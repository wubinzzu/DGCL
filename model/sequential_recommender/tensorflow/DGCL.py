from model.base import AbstractRecommender
from model.base.abstract_recommender import SocialAbstractRecommender
import tensorflow as tf
from util.tensorflow.loss import log_loss
import numpy as np
from util.tensorflow import l2_distance
from collections import defaultdict
from util.tensorflow import l2_loss, get_session
import scipy.sparse as sp
from reckit import timer
from reckit import pad_sequences
# reckit==0.2.4
from data.sampler import TimeOrderPairwiseSampler
import os
import math
from util.tensorflow.func import sp_mat_to_sp_tensor, normalize_adj_matrix
from util.common.tool import csr_to_user_dict_bytime, csr_to_time_dict, csr_to_user_dict
import random


def mexp(x, tau=1.0):
    # normalize att_logit to avoid negative value
    x_max = tf.reduce_max(x)
    x_min = tf.reduce_min(x)
    norm_x = (x-x_min) / (x_max-x_min)
    exp_x = tf.exp(norm_x/tau)
    return exp_x


class DGCL(SocialAbstractRecommender):
    def __init__(self, config):
        super(DGCL, self).__init__(config)
        self.config = config
        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.reg_w = config["reg_w"]
        self.reg_cl = config["reg_cl"]
        self.n_layers_ii = config["n_layers_ii"]
        self.n_layers_uu = config["n_layers_uu"]
        self.n_layers_ui = config["n_layers_ui"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.n_seqs = config["n_seqs"]
        self.n_next = config["n_next"]
        self.n_next_neg = config["n_next_neg"]
        self.temp = config["temp"]
        self.scope = config["scope"]

        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.train_data.to_user_dict(by_time=True)
        self.user_pos_time = csr_to_time_dict(self.dataset.time_matrix)  # pos time
        self.norm_adj_social = self._create_social_adj_mat()
        self.norm_adj_ui = self._create_ui_adj_mat()
        self.socialDict = self._get_SocialDict()
        self.all_users = list(self.user_pos_train.keys())
        # self.friends_list = self._get_friends_list()
        self._process_test()  # 生成用户交互的物品序列
        self.sess = get_session(config["gpu_mem"])
        self._build_model()
        self.sess.run(tf.global_variables_initializer())
        self.global_step = -1

    def _create_ui_adj_mat(self):
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        n_nodes = self.users_num + self.items_num
        up_left_adj = sp.csr_matrix((ratings, (users_np, items_np+self.users_num)), shape=(n_nodes, n_nodes))
        adj_mat = up_left_adj + up_left_adj.T
        adj_matrix = normalize_adj_matrix(adj_mat, norm_method="symmetric")
        return adj_matrix

    def _create_social_adj_mat(self):
        # 用于生成用户社交临接矩阵
        uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
        u1_idx, u2_idx = list(zip(*uu_idx))

        self.u1_idx = tf.constant(u1_idx, dtype=tf.int32, shape=None, name="u1_idx")
        self.u2_idx = tf.constant(u2_idx, dtype=tf.int32, shape=None, name="u2_idx")

        u1_idx = np.array(u1_idx, dtype=np.int32)
        u2_idx = np.array(u2_idx, dtype=np.int32)
        ratings = np.ones_like(u1_idx, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (u1_idx, u2_idx)),
                                shape=(self.users_num, self.users_num))
        adj_mat = tmp_adj + tmp_adj.T
        return self._normalize_spmat(adj_mat)

    def _normalize_spmat(self, adj_mat):
        # pre adjcency matrix（对上面用户社交临接矩阵处理）
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        print('use the pre adjcency matrix')
        return adj_matrix

    def _split_A_att_hat(self, X, embedding):
        A_fold_hat = []
        start = 0
        end = self.users_num
        A_fold_hat.append(self._convert_sp_mat_to_sp_att_tensor(X[start:end], start, embedding))
        return A_fold_hat

    def _convert_sp_mat_to_sp_att_tensor(self, X, start, embedding):
        coo = X.tocoo().astype(np.float32)
        l2_embedding = tf.nn.l2_normalize(embedding, dim=1)
        indices = np.mat([coo.row, coo.col]).transpose()
        center = tf.nn.embedding_lookup(l2_embedding, coo.row + start)
        neighbor = tf.nn.embedding_lookup(l2_embedding, coo.col)
        attention = tf.reduce_sum(tf.multiply(center, neighbor), axis=1, keepdims=False)
        dim_1 = tf.sqrt(tf.reduce_sum(tf.square(center), axis=1))
        dim_2 = tf.sqrt(tf.reduce_sum(tf.square(neighbor), axis=1))
        attention = attention/tf.multiply(dim_1, dim_2)
        sp_mat = tf.sparse_softmax(tf.SparseTensor(indices, attention, coo.shape))
        return sp_mat

    def _process_test(self):
        # 生成用户交互的物品序列
        item_seqs = [self.user_pos_train[user][-self.n_seqs:] if user in self.user_pos_train else [self.items_num]
                     for user in range(self.users_num)]
        self.test_item_seqs = pad_sequences(item_seqs, value=self.items_num, max_len=self.n_seqs,
                                            padding='pre', truncating='pre', dtype=np.int32)

    def _create_placeholder(self):
        # 创建占位符
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.user_social_ph = tf.placeholder(tf.int32, [None, None], name="friends")  # 用户的社交关系
        self.head_ph = tf.placeholder(tf.int32, [None, self.n_seqs], name="head_item")  # the previous item
        self.pos_tail_ph = tf.placeholder(tf.int32, [None, self.n_next], name="pos_tail_item")  # the next item
        self.neg_tail_ph = tf.placeholder(tf.int32, [None, self.n_next_neg], name="neg_tail_item")  # the negative item

    def _construct_graph(self):
        # 生成了用户序列的时间间隔，计算方式：math.log((每两个物品的时间间隔/最小时间间隔)+1) * self.time_mul
        th_rs_dict = defaultdict(list)
        th_rs_dict_time = defaultdict(list)
        for user, pos_items in self.user_pos_train.items():
            seq_times = self.user_pos_time[user]
            seq_times = sorted(seq_times)
            seq_times_set = sorted(set(seq_times))
            seq_times_inter = []
            time_min = 1
            if len(seq_times_set) > 1:
                for i in range(len(seq_times_set) - 1):
                    time_min_tmp = seq_times_set[i + 1] - seq_times_set[i]
                    seq_times_inter.append(time_min_tmp)
                time_min = sorted(seq_times_inter)[0]
            time_i = 0
            for h, t in zip(pos_items[:-1], pos_items[1:]):
                # time_ii = math.log((seq_times[time_i + 1] - seq_times[time_i]) / 3600 + 1) * self.time_mul
                time_ii = math.log((seq_times[time_i + 1] - seq_times[time_i]) / time_min + 1)
                th_rs_dict[(t, h)].append(user)  # (头，用户，尾)三元组
                th_rs_dict_time[(t, h)].append(time_ii)
                time_i += 1

        th_rs_list = sorted(th_rs_dict.items(), key=lambda x: x[0])
        th_rs_list_time = sorted(th_rs_dict_time.items(), key=lambda x: x[0])

        user_list, head_list, tail_list, time_distance_list = [], [], [], []
        for (t, h), r in th_rs_list:
            user_list.extend(r)
            head_list.extend([h] * len(r))  # 所有头结点
            tail_list.extend([t] * len(r))  # 所有尾节点
        for (t, h), time in th_rs_list_time:
            time_distance_list.extend(time)  # 添加成对物品之间计算后的时间间隔得分

        # attention mechanism
        # the auxiliary constant to calculate softmax
        row_idx, nnz = np.unique(tail_list, return_counts=True)
        count = {r: n for r, n in zip(row_idx, nnz)}
        nnz = [count[i] if i in count else 0 for i in range(self.items_num)]
        nnz = np.concatenate([[0], nnz])
        rows_idx = np.cumsum(nnz)

        # the auxiliary constant to calculate the weight between two node
        edge_num = np.array([len(r) for (t, h), r in th_rs_list], dtype=np.int32)
        edge_num = np.concatenate([[0], edge_num])
        edge_idx = np.cumsum(edge_num)

        sp_idx = [[t, h] for (t, h), r in th_rs_list]
        adj_mean_norm = self._get_mean_norm(edge_num[1:], sp_idx)

        return head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_mean_norm, time_distance_list

    @timer
    def _init_constant(self):
        # 生成张量
        head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_norm, time_distance_list = self._construct_graph()

        # attention mechanism
        self.att_head_idx = tf.constant(head_list, dtype=tf.int32, shape=None, name="att_head_idx")
        self.att_tail_idx = tf.constant(tail_list, dtype=tf.int32, shape=None, name="att_tail_idx")
        self.att_user_idx = tf.constant(user_list, dtype=tf.int32, shape=None, name="att_user_idx")
        self.att_time_idx = tf.constant(time_distance_list, dtype=float)
        self.att_time_idx = tf.expand_dims(self.att_time_idx, 1)
        # the auxiliary constant to calculate softmax
        self.rows_end_idx = tf.constant(rows_idx[1:], dtype=tf.int32, shape=None, name="rows_end_idx")
        self.row_begin_idx = tf.constant(rows_idx[:-1], dtype=tf.int32, shape=None, name="row_begin_idx")

        # the auxiliary constant to calculate the weight between two node
        self.edge_end_idx = tf.constant(edge_idx[1:], dtype=tf.int32, shape=None, name="edge_end_idx")
        self.edge_begin_idx = tf.constant(edge_idx[:-1], dtype=tf.int32, shape=None, name="edge_begin_idx")

        # the index of sparse matrix
        self.sp_tensor_idx = tf.constant(sp_idx, dtype=tf.int64)
        self.adj_norm = None

    def _get_mean_norm(self, edge_num, sp_idx):
        adj_num = np.array(edge_num, dtype=np.float32)
        rows, cols = list(zip(*sp_idx))
        adj_mat = sp.csr_matrix((adj_num, (rows, cols)), shape=(self.items_num, self.items_num))

        return normalize_adj_matrix(adj_mat, "left").astype(np.float32)

    def _init_variable(self):
        with tf.name_scope("Embeddings"):
            self.embeddings = dict()
            # 权重
            init = tf.random.truncated_normal([self.factors_num, self.factors_num], mean=0.0, stddev=0.01)
            self.weight_social_trans = tf.Variable(init, dtype=tf.float32, name="weight_social_trans")
            self.weight_item_trans = tf.Variable(init, dtype=tf.float32, name="weight_item_trans")
            self.weight_social_transB = tf.Variable(tf.constant(0.00, shape=[self.factors_num]),
                                                    name="weight_social_transB")
            self.weight_item_transB = tf.Variable(tf.constant(0.00, shape=[self.factors_num]),
                                                  name="weight_item_transB")
            self.weight_user_attFuse = tf.Variable(init, dtype=tf.float32, name="weight_user_attFuse")
            self.weight_user_attFuseB = tf.Variable(tf.constant(0.00, shape=[self.factors_num]),
                                                    name="weight_user_attFuseB")
            self.weight_item_attFuse = tf.Variable(init, dtype=tf.float32, name="weight_item_attFuse")
            self.weight_item_attFuseB = tf.Variable(tf.constant(0.00, shape=[self.factors_num]),
                                                    name="weight_item_attFuseB")
            self.weight_user_attFuse_ui = tf.Variable(init, dtype=tf.float32, name="weight_user_attFuse_ui")
            self.weight_user_attFuseB_ui = tf.Variable(tf.constant(0.00, shape=[self.factors_num]),
                                                    name="weight_user_attFuseB_ui")
            self.weight_item_attFuse_ui = tf.Variable(init, dtype=tf.float32, name="weight_item_attFuse_ui")
            self.weight_item_attFuseB_ui = tf.Variable(tf.constant(0.00, shape=[self.factors_num]),
                                                    name="weight_item_attFuseB_ui")

            init = tf.random.truncated_normal([1, self.factors_num], mean=0.0, stddev=0.01)
            self.weight_time = tf.Variable(init, dtype=tf.float32, name="weight_time")
            self.weight_timeB = tf.Variable(tf.constant(0.00, shape=[self.factors_num]), name="weight_timeB")

            # 用户
            init = tf.random.truncated_normal([self.users_num, self.factors_num], mean=0.0, stddev=0.01)
            user_embeddings = tf.Variable(init, dtype=tf.float32)
            self.embeddings.setdefault("user_embeddings", user_embeddings)
            user_embeddings_social = tf.Variable(init, dtype=tf.float32)
            self.embeddings.setdefault("user_embeddings_social", user_embeddings)
            user_embeddings_social_gcn = self._social_gcn(user_embeddings_social)
            # 物品
            init = tf.random.truncated_normal([self.items_num, self.factors_num], mean=0.0, stddev=0.01)
            item_embeddings = tf.Variable(init, dtype=tf.float32)
            self.embeddings.setdefault("item_embeddings", item_embeddings)
            item_embeddings_ii = tf.Variable(init, dtype=tf.float32)
            self.embeddings.setdefault("item_embeddings_ii", item_embeddings_ii)
            self.item_biases = tf.Variable(tf.zeros([self.items_num]), dtype=tf.float32, name="item_biases")
            self.end = tf.constant(0.0, tf.float32, [1, self.factors_num], name='end')
            user_embeddings_gcn, item_embeddings_gcn = self._ui_gcn(user_embeddings, item_embeddings)
            item_embeddings_ii_gcn = self._item_gcn(item_embeddings_ii, user_embeddings_gcn)

            self.user_embeddings = tf.concat([user_embeddings_gcn, self.end], 0)
            self.user_embeddings_social = tf.concat([user_embeddings_social_gcn, self.end], 0)
            self.item_embeddings = tf.concat([item_embeddings_gcn, self.end], 0)
            self.item_embeddings_ii = tf.concat([item_embeddings_ii_gcn, self.end], 0)

    def _ui_gcn(self, user_emb, item_emb):
        with tf.name_scope("ui_gcn"):
            norm_adj = self.norm_adj_ui
            adj_mat = sp_mat_to_sp_tensor(norm_adj)
            ego_embeddings = tf.concat([user_emb, item_emb], axis=0)
            all_embeddings = [ego_embeddings]
            for k in range(0, self.n_layers_ui):
                ego_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings)
                all_embeddings += [ego_embeddings]

            all_embeddings = tf.stack(all_embeddings, 1)
            all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
            u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.users_num, self.items_num], 0)
            return u_g_embeddings, i_g_embeddings

    def _social_gcn(self, user_emb):
        with tf.name_scope("social_gcn"):
            # 社交GAT
            ego_embeddings = user_emb
            all_embeddings = [ego_embeddings]
            for k in range(0, self.n_layers_uu):
                A_fold_hat = self._split_A_att_hat(self.norm_adj_social, all_embeddings[-1])
                temp_embed = []
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[0], ego_embeddings))
                # sum messages of neighbors.
                side_embeddings = tf.concat(temp_embed, 0)
                # transformed sum messages of neighbors.
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = tf.stack(all_embeddings, 1)
            all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
            return all_embeddings

    def _item_gcn(self, item_emb_ii, user_emb):
        with tf.name_scope("item_gcn"):
            item_emb = item_emb_ii
            for k in range(self.n_layers_ii):
                att_scores = self._get_attention(item_emb, user_emb)
                neighbor_embeddings = tf.sparse_tensor_dense_matmul(att_scores, item_emb)
                item_emb = item_emb + neighbor_embeddings
            return item_emb

    def _get_attention(self, item_embeddings, user_embeddings):
        # 加入时间间隔后计算attention分数
        h_embed = tf.nn.embedding_lookup(item_embeddings, self.att_head_idx)
        r_embed = tf.nn.embedding_lookup(user_embeddings, self.att_user_idx)
        t_embed = tf.nn.embedding_lookup(item_embeddings, self.att_tail_idx)
        time_embed = tf.nn.tanh(tf.matmul(self.att_time_idx, self.weight_time) + self.weight_timeB)
        att_logit = l2_distance(h_embed+r_embed+time_embed, t_embed)
        exp_logit = mexp(-att_logit, 1.0)
        exp_logit = tf.concat([[0], exp_logit], axis=0)
        sum_exp_logit = tf.cumsum(exp_logit)
        pre_sum = tf.gather(sum_exp_logit, self.edge_begin_idx)
        next_sum = tf.gather(sum_exp_logit, self.edge_end_idx)
        sum_exp_logit_per_edge = next_sum - pre_sum
        # convert to spares tensor
        exp_logit = tf.SparseTensor(indices=self.sp_tensor_idx, values=sum_exp_logit_per_edge,
                                    dense_shape=[self.items_num, self.items_num])
        # normalize attention score to a probability vector
        next_sum = tf.gather(sum_exp_logit, self.rows_end_idx)
        pre_sum = tf.gather(sum_exp_logit, self.row_begin_idx)
        sum_exp_logit_per_row = next_sum - pre_sum + 1e-6
        sum_exp_logit_per_row = tf.reshape(sum_exp_logit_per_row, shape=[self.items_num, 1])
        att_score = exp_logit / sum_exp_logit_per_row
        return att_score

    def _forward_head_emb(self,  item_embeddings):
        # embed item sequence
        item_seq_embs = tf.nn.embedding_lookup(item_embeddings, self.head_ph)  # (b,l,d)
        mask = tf.cast(tf.not_equal(self.head_ph, self.items_num), dtype=tf.float32)  # (b,l)
        his_emb = tf.reduce_sum(item_seq_embs, axis=1) / tf.reduce_sum(mask, axis=1, keepdims=True)  # (b,d)/(b,1)
        head_emb_g = tf.nn.embedding_lookup(item_embeddings, self.head_ph[:, -1])  # b*d
        head_emb = head_emb_g + his_emb
        return head_emb

    def _get_SocialDict(self):
        SocialDict = {}
        for u in range(self.users_num):
            trustors = self.social_matrix[u].indices
            if len(trustors)>0:
                SocialDict[u] = trustors.tolist()
            else:
                SocialDict[u] = [self.users_num]
        return SocialDict

    def _pre_att_transfer(self, user_emb, uu_emb, ii_emb):
        with tf.name_scope("pre_trans"):
            uu_w = tf.nn.relu(tf.matmul(uu_emb,self.weight_social_trans)+self.weight_social_transB)
            uu_w = tf.exp(tf.reduce_mean(tf.multiply(uu_w, user_emb), axis=1, keepdims=True))
            ii_w = tf.nn.relu(tf.matmul(ii_emb,self.weight_item_trans)+self.weight_item_transB)
            ii_w = tf.exp(tf.reduce_mean(tf.multiply(ii_w, user_emb), axis=1, keepdims=True))
            epsilon = 1e-9
            seq_w = tf.div(ii_w, ii_w+uu_w+epsilon)
            social_w = 1.0 - seq_w
            # 截断
            if self.scope == 1:
                social_w = tf.clip_by_value(social_w, 0.0, 1e-6)
                seq_w = tf.clip_by_value(seq_w, (1 - 1e-6), 1.0)
                pre_emb = user_emb + social_w * uu_emb + seq_w * ii_emb
            else:
                social_w = tf.clip_by_value(social_w, 0.8, 0.85)
                seq_w = tf.clip_by_value(seq_w, 0.15, 0.2)
                pre_emb = social_w * uu_emb + seq_w * ii_emb
            # pre_emb = user_emb + social_w * uu_emb + seq_w * ii_emb
            return pre_emb

    def _attFuse_user_friends(self, user_embeddings, weight, weight_b):
        with tf.name_scope("att_fuse_user"):
            user_emb = tf.nn.embedding_lookup(user_embeddings, self.user_ph)
            friends_emb_user = tf.nn.embedding_lookup(user_embeddings, self.user_social_ph)
            epsilon = 1e-9
            n = tf.shape(friends_emb_user)[1]
            user_emb = tf.expand_dims(user_emb, axis=1)
            user_emb = tf.tile(user_emb, tf.stack([1, n, 1]))
            mlp_put = tf.nn.relu(tf.multiply(user_emb, tf.matmul(friends_emb_user, weight, transpose_b=True)+weight_b))
            A = tf.reduce_mean(mlp_put, -1)
            exp_A = tf.exp(A)
            mask_mat = tf.to_float(tf.not_equal(self.user_social_ph, self.users_num))
            exp_A = mask_mat * exp_A
            exp_sum = tf.reduce_sum(exp_A, 1, keepdims=True)
            exp_sum = tf.pow(exp_sum + epsilon, tf.constant(1, tf.float32, [1]))
            A_ = tf.expand_dims(tf.div(exp_A, exp_sum) * mask_mat, 2)
            return tf.reduce_sum(A_ * friends_emb_user, 1, keepdims=False)

    def _attFuse_item_seqs(self, user_emb, item_emb, weight, weight_b):
        with tf.name_scope("att_fuse_item"):
            epsilon = 1e-9
            user_emb = tf.expand_dims(user_emb, axis=1)
            user_emb = tf.tile(user_emb, tf.stack([1, self.n_seqs, 1]))
            mlp_put = tf.nn.relu(tf.multiply(user_emb, tf.matmul(item_emb, weight, transpose_b=True) + weight_b))
            A = tf.reduce_mean(mlp_put, -1)
            exp_A = tf.exp(A)
            mask_mat = tf.to_float(tf.not_equal(self.head_ph, self.items_num))
            exp_A = mask_mat * exp_A
            exp_sum = tf.reduce_sum(exp_A, 1, keepdims=True)
            exp_sum = tf.pow(exp_sum + epsilon, tf.constant(1, tf.float32, [1]))
            A_ = tf.expand_dims(tf.div(exp_A, exp_sum) * mask_mat, 2)
            return tf.reduce_sum(A_ * item_emb, 1, keepdims=False)

    def InfoNCE(self, view1, view2, temperature):
        # 对比学习损失
        view1 = tf.nn.l2_normalize(view1, dim=1)
        view2 = tf.nn.l2_normalize(view2, dim=1)
        pos_score = tf.reduce_sum(view1 * view2, axis=-1)
        pos_score = tf.exp(pos_score / temperature)
        ttl_score = tf.matmul(view1, tf.transpose(view2))
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / temperature), axis=1)
        cl_loss = -tf.log(pos_score / ttl_score)
        return tf.reduce_mean(cl_loss)

    def _build_model(self):
        self._create_placeholder()
        self._init_constant()
        self._init_variable()
        # Translation-based Recommendation
        social_emb = tf.nn.embedding_lookup(self.user_embeddings_social, self.user_ph)  # b*d
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # b*d
        head_emb = self._forward_head_emb(self.item_embeddings_ii)
        seq_emb = tf.nn.embedding_lookup(self.item_embeddings_ii, self.head_ph)
        item_emb_ui = tf.nn.embedding_lookup(self.item_embeddings, self.head_ph)
        user_att_friends = self._attFuse_user_friends(self.user_embeddings, self.weight_user_attFuse_ui, self.weight_user_attFuseB_ui)
        social_att_friends = self._attFuse_user_friends(self.user_embeddings_social, self.weight_user_attFuse, self.weight_user_attFuseB)
        item_att_l = self._attFuse_item_seqs(user_emb, item_emb_ui, self.weight_item_attFuse_ui, self.weight_item_attFuseB_ui)
        seq_att_l = self._attFuse_item_seqs(user_emb, seq_emb, self.weight_item_attFuse, self.weight_item_attFuseB)

        pos_tail_emb = tf.nn.embedding_lookup(self.item_embeddings_ii, self.pos_tail_ph)  # b*t*d
        neg_tail_emb = tf.nn.embedding_lookup(self.item_embeddings_ii, self.neg_tail_ph)  # b*t*d

        pos_tail_bias = tf.gather(self.item_biases, self.pos_tail_ph)  # b*t
        neg_tail_bias = tf.gather(self.item_biases, self.neg_tail_ph)  # b*t

        # 权重
        # pre_emb = self._pre_att_transfer(user_emb, social_emb, head_emb)
        pre_emb = self._pre_att_transfer(user_emb, social_emb, seq_att_l)
        pre_emb = tf.expand_dims(pre_emb, axis=1)
        pos_rating = -l2_distance(pre_emb, pos_tail_emb) + pos_tail_bias  # b*t
        neg_rating = -l2_distance(pre_emb, neg_tail_emb) + neg_tail_bias  # b*t

        pairwise_loss = tf.reduce_sum(log_loss(pos_rating - neg_rating))

        # reg loss
        emb_reg = l2_loss(user_emb, social_emb, head_emb, pos_tail_emb, neg_tail_emb, pos_tail_bias, neg_tail_bias)
        # weight loss
        weight_reg = tf.reduce_sum(tf.square(self.weight_social_trans)) + tf.reduce_sum(tf.square(self.weight_item_trans)) + \
                     tf.reduce_sum(tf.square(self.weight_social_transB)) + tf.reduce_sum(tf.square(self.weight_item_transB)) + \
                     tf.reduce_sum(tf.square(self.weight_user_attFuse)) + tf.reduce_sum(tf.square(self.weight_user_attFuseB)) + \
                     tf.reduce_sum(tf.square(self.weight_user_attFuse_ui)) + tf.reduce_sum(tf.square(self.weight_user_attFuseB_ui)) + \
                     tf.reduce_sum(tf.square(self.weight_item_attFuse_ui)) + tf.reduce_sum(tf.square(self.weight_item_attFuseB_ui)) + \
                     tf.reduce_sum(tf.square(self.weight_item_attFuse)) + tf.reduce_sum(tf.square(self.weight_item_attFuseB)) + \
                     tf.reduce_sum(tf.square(self.weight_time)) + tf.reduce_sum(tf.square(self.weight_timeB))
        # contrastive loss  (infoNCE版本)
        cl1_loss = self.InfoNCE(social_att_friends, user_att_friends, self.temp)
        cl2_loss = self.InfoNCE(seq_att_l, item_att_l, self.temp)
        cl_loss = cl1_loss + cl2_loss

        # objective loss and optimizer
        obj_loss = pairwise_loss + self.reg * emb_reg + self.reg_cl * cl_loss + self.reg_w * weight_reg
        self.update_opt = tf.train.AdamOptimizer(self.lr).minimize(obj_loss)

        # for prediction
        self.item_embeddings_final = tf.Variable(tf.zeros([self.items_num, self.factors_num]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.assign_opt = tf.assign(self.item_embeddings_final, self.item_embeddings_ii[:self.items_num, :])

        j_emb = tf.expand_dims(self.item_embeddings_final, axis=0) # 1*n*d
        self.prediction = -l2_distance(pre_emb, j_emb) + self.item_biases  # b*n

    def train_model(self):
        # a = self.sess.run(self.a)
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data, len_seqs=self.n_seqs, len_next=self.n_next,
                                             pad=self.items_num, num_neg=self.n_next_neg,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=False)
        bets_result = 0.0
        counter = 0
        best_str = ""
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.global_step+1, self.epochs):
            for bat_users, bat_head, bat_pos_tail, bat_neg_tail in data_iter:
                # bat_head[np.where(bat_head == self.items_num)] = self.items_num - 1
                bat_social = list()
                for u in bat_users:
                    bat_social.append(self.socialDict[u])
                bat_social = pad_sequences(bat_social, value=self.users_num)
                feed = {self.user_ph: bat_users,
                        self.user_social_ph: bat_social,
                        self.head_ph: bat_head.reshape([-1, self.n_seqs]),
                        self.pos_tail_ph: bat_pos_tail.reshape([-1, self.n_next]),
                        self.neg_tail_ph: bat_neg_tail.reshape([-1, self.n_next_neg]),
                        }
                self.sess.run(self.update_opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))
            counter += 1
            if counter > 50:
                self.logger.info("early stop")
                break
            cur_result = float(result.split("\t")[1])
            if cur_result >= bets_result:
                bets_result = cur_result
                best_str = result
                counter = 0
        self.logger.info("best:\t%s" % best_str)

    def evaluate_model(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, users):
        last_items = [self.test_item_seqs[u] for u in users]
        bat_social = list()
        for u in users:
            bat_social.append(self.socialDict[u])
        bat_social = pad_sequences(bat_social, value=self.users_num)
        feed = {self.user_ph: users, self.user_social_ph: bat_social, self.head_ph: last_items}
        bat_ratings = self.sess.run(self.prediction, feed_dict=feed)
        return bat_ratings
