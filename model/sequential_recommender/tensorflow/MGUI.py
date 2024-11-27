from model.base import AbstractRecommender
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
epsilon = 1e-9

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


def mexp(x, tau=1.0):
    # normalize att_logit to avoid negative value
    x_max = tf.reduce_max(x)
    x_min = tf.reduce_min(x)
    norm_x = (x-x_min) / (x_max-x_min)
    # calculate attention for each pair of items
    # used for calculating softmax
    exp_x = tf.exp(norm_x/tau)
    return exp_x


class MGUI(AbstractRecommender):
    def __init__(self, config):
        super(MGUI, self).__init__(config)
        self.config = config
        self.factors_num = config["factors_num"]
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.n_layers_ii = config["n_layers_ii"]
        self.n_layers_ui = config["n_layers_ui"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.n_seqs = config["n_seqs"]
        self.n_max_length = config["n_max_length"]
        self.n_next = config["n_next"]
        self.n_next_neg = config["n_next_neg"]
        self.num_outputs_secondCaps_multi = config["num_outputs_secondCaps_multi"]
        self.tau = config["tau"]
        self.blocks = config["blocks"]
        self.heads = config["heads"]
        self.dropout_rate = config["dropout_rate"]

        self.users_num, self.items_num = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.train_data.to_user_dict(by_time=True)
        self.user_pos_time = csr_to_time_dict(self.dataset.time_matrix)
        self.norm_adj_ui = self._create_ui_adj_mat()
        self.all_users = list(self.user_pos_train.keys())
        self._process_test()  # 生成用户交互的物品序列
        self.sess = get_session(config["gpu_mem"])
        self._build_model()
        self.sess.run(tf.global_variables_initializer())
        # restore model
        self.save_interval = config["save_interval"]
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

    def _process_test(self):
        # 生成用户交互的物品序列
        item_seqs = [self.user_pos_train[user][-self.n_seqs:] if user in self.user_pos_train else [self.items_num-1]
                     for user in range(self.users_num)]
        self.test_item_seqs = pad_sequences(item_seqs, value=self.items_num-1, max_len=self.n_max_length,
                                            padding='pre', truncating='pre', dtype=np.int32)

    def _create_placeholder(self):
        # 创建占位符
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.max_ph = tf.placeholder(tf.int32, [None, self.n_max_length], name="max_item")  # 用于胶囊网络
        self.head_ph = tf.placeholder(tf.int32, [None, self.n_seqs], name="head_item")  # the previous item
        self.pos_tail_ph = tf.placeholder(tf.int32, [None, self.n_next], name="pos_tail_item")  # the next item
        self.neg_tail_ph = tf.placeholder(tf.int32, [None, self.n_next_neg], name="neg_tail_item")  # the negative item
        self.is_training = tf.placeholder(tf.bool, name="training_flag")
        self.timenow_ph = tf.compat.v1.placeholder(tf.float32, [None, self.n_seqs], name="time_now")

    def _construct_graph(self):
        th_rs_dict = defaultdict(list)
        for user, pos_items in self.user_pos_train.items():
            for h, t in zip(pos_items[:-1], pos_items[1:]):
                th_rs_dict[(t, h)].append(user)  # (头，用户，尾)三元组
        th_rs_list = sorted(th_rs_dict.items(), key=lambda x: x[0])
        user_list, head_list, tail_list = [], [], []
        for (t, h), r in th_rs_list:
            user_list.extend(r)
            head_list.extend([h] * len(r))  # 所有头结点
            tail_list.extend([t] * len(r))  # 所有尾节点
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
        return head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_mean_norm

    @timer
    def _init_constant(self):
        # 生成张量
        head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_norm = self._construct_graph()

        # attention mechanism
        self.att_head_idx = tf.constant(head_list, dtype=tf.int32, shape=None, name="att_head_idx")
        self.att_tail_idx = tf.constant(tail_list, dtype=tf.int32, shape=None, name="att_tail_idx")
        self.att_user_idx = tf.constant(user_list, dtype=tf.int32, shape=None, name="att_user_idx")

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
        # embedding parameters
        self.embeddings = dict()
        init = tf.random.truncated_normal([self.users_num, self.factors_num], mean=0.0, stddev=0.01)
        user_embeddings = tf.Variable(init, dtype=tf.float32)
        self.embeddings.setdefault("user_embeddings", user_embeddings)

        init = tf.random.truncated_normal([self.items_num, self.factors_num], mean=0.0, stddev=0.01)
        item_embeddings = tf.Variable(init, dtype=tf.float32)
        self.embeddings.setdefault("item_embeddings", item_embeddings)
        self.end = tf.constant(0.0, tf.float32, [1, self.factors_num], name='end')
        user_embeddings_gcn, item_embeddings_gcn = self._ui_gcn(user_embeddings, item_embeddings)
        item_embeddings_ii_gcn = self._item_gcn(item_embeddings, user_embeddings_gcn)
        self.user_embeddings = tf.concat([user_embeddings_gcn, self.end], 0)
        self.item_embeddings = tf.concat([item_embeddings_gcn, self.end], 0)
        self.item_embeddings_ii = tf.concat([item_embeddings_ii_gcn, self.end], 0)
        self.item_biases = tf.Variable(tf.zeros([self.items_num]), dtype=tf.float32, name="item_biases")

        init = tf.random.truncated_normal([self.factors_num, self.factors_num], mean=0.0, stddev=0.01)
        self.item_embeddings_weight = tf.Variable(init, dtype=tf.float32, name="item_embeddings_weight")
        self.weight_oo = tf.Variable(init, dtype=tf.float32, name="weight_oo")
        self.weight_mm = tf.Variable(init, dtype=tf.float32, name="weight_mm")
        self.weight_graph = tf.Variable(init, dtype=tf.float32, name="weight_graph")
        self.weight_time = tf.Variable(init, dtype=tf.float32, name="weight_time")

        init = tf.random.truncated_normal([1, self.n_max_length, self.num_outputs_secondCaps_multi*self.factors_num, self.factors_num], mean=0.0, stddev=0.01)
        self.routing_W = tf.Variable(init, dtype=tf.float32, name="routing_W")

    def _ui_gcn(self, user_emb, item_emb):
        adj_mat = sp_mat_to_sp_tensor(self.norm_adj_ui)
        ego_embeddings = tf.concat([user_emb, item_emb], axis=0)
        all_embeddings = [ego_embeddings]
        for k in range(0, self.n_layers_ui):
            ego_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.users_num, self.items_num], 0)
        return u_g_embeddings, i_g_embeddings

    def _get_attention(self, item_embeddings, user_embeddings):
        h_embed = tf.nn.embedding_lookup(item_embeddings, self.att_head_idx)
        r_embed = tf.nn.embedding_lookup(user_embeddings, self.att_user_idx)
        t_embed = tf.nn.embedding_lookup(item_embeddings, self.att_tail_idx)
        att_logit = l2_distance(h_embed+r_embed, t_embed)
        exp_logit = mexp(-att_logit, self.tau)
        exp_logit = tf.concat([[0], exp_logit], axis=0)
        sum_exp_logit = tf.cumsum(exp_logit)
        pre_sum = tf.gather(sum_exp_logit, self.edge_begin_idx)
        next_sum = tf.gather(sum_exp_logit, self.edge_end_idx)
        sum_exp_logit_per_edge = next_sum - pre_sum
        exp_logit = tf.SparseTensor(indices=self.sp_tensor_idx, values=sum_exp_logit_per_edge,
                                    dense_shape=[self.items_num, self.items_num])
        next_sum = tf.gather(sum_exp_logit, self.rows_end_idx)
        pre_sum = tf.gather(sum_exp_logit, self.row_begin_idx)
        sum_exp_logit_per_row = next_sum - pre_sum + 1e-6
        sum_exp_logit_per_row = tf.reshape(sum_exp_logit_per_row, shape=[self.items_num, 1])
        att_score = exp_logit / sum_exp_logit_per_row
        return att_score

    def _item_gcn(self, item_emb, user_emb):
        with tf.name_scope("item_gcn"):
            for k in range(self.n_layers_ii):
                att_scores = self._get_attention(item_emb, user_emb)
                neighbor_embeddings = tf.sparse_tensor_dense_matmul(att_scores, item_emb)
                item_emb = item_emb + neighbor_embeddings
            return item_emb

    def CapsLayer(self, input, layer_type, kernel_size=None, stride=None):
        u = input
        item_emb_hat = tf.reduce_sum(self.routing_W[:,:self.n_max_length, :, :]*u, axis=3)
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.n_max_length, self.num_outputs_secondCaps_multi, self.factors_num])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_outputs_secondCaps_multi, self.n_max_length, self.factors_num])
        item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        capsule_weight = tf.stop_gradient(tf.zeros([self.batch_size_b, self.num_outputs_secondCaps_multi, self.n_max_length]))
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.max_ph, self.items_num)), 1)
        for i in range(3):
            attn_mask = tf.tile(mask, [1, self.num_outputs_secondCaps_multi, 1])
            paddings = tf.zeros_like(attn_mask)
            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(attn_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)
            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule
                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_outputs_secondCaps_multi, self.n_max_length])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule
        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_outputs_secondCaps_multi, self.factors_num])
        return interest_capsule

    def Multi_Multi(self, item_emb):
        self.batch_size_b = tf.shape(item_emb)[0]
        item_input = tf.expand_dims(item_emb, axis=2)
        with tf.variable_scope('item_SecondCaps_layer'):
            self.secondCaps_multi = self.CapsLayer(item_input, layer_type='item')
        item_emb_multi = self.secondCaps_multi
        reuse = None
        with tf.variable_scope("SASRec", reuse=reuse):
            item_emb_multi = tf.layers.dropout(item_emb_multi,
                                         rate=self.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            for i in range(self.blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    item_emb_multi = multihead_attention(queries=normalize(item_emb_multi),
                                                   keys=item_emb_multi,
                                                   num_units=self.factors_num,
                                                   num_heads=self.heads,
                                                   dropout_rate=self.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")
                    item_emb_multi = feedforward(normalize(item_emb_multi),
                                           num_units=[self.factors_num, self.factors_num],
                                           dropout_rate=self.dropout_rate,
                                           is_training=self.is_training)

            item_emb_multi = normalize(item_emb_multi)  # (b, l, d)
        return item_emb_multi

    def _One_one(self, item_emb, item_emb_weight):
        time = tf.tile(tf.expand_dims(self.timenow_ph, -1), tf.stack([1, 1, self.factors_num]))  # b,L,d
        time_variable = tf.matmul(time, self.weight_time)
        item_emb = item_emb + time_variable
        item_emb_gate = tf.matmul(item_emb, item_emb_weight)
        item_emb_gate = tf.sigmoid(item_emb_gate)
        item_emb_gate = tf.multiply(item_emb_gate, item_emb) + item_emb
        return item_emb_gate

    def _forward_head_emb(self,  item_embeddings):
        item_seq_embs = tf.nn.embedding_lookup(item_embeddings, self.head_ph)  # (b,l,d)
        mask = tf.cast(tf.not_equal(self.head_ph, self.items_num), dtype=tf.float32)  # (b,l)
        his_emb = tf.reduce_sum(item_seq_embs, axis=1) / tf.reduce_sum(mask, axis=1, keepdims=True)  # (b,d)/(b,1)
        head_emb_g = tf.nn.embedding_lookup(item_embeddings, self.head_ph[:, -1])  # b*d
        head_emb = head_emb_g + his_emb
        return head_emb

    def _fusion_gate(self, emb_oo, emb_mm, emb_seq, emb_user):
        emb_mm = tf.multiply(emb_mm, emb_oo)
        emb_oo_gate = tf.matmul(emb_oo, self.weight_oo)
        emb_oo_gate = tf.multiply(tf.sigmoid(emb_oo_gate), emb_oo)
        emb_mm_gate = tf.matmul(emb_mm, self.weight_mm)
        emb_mm_gate = tf.multiply(tf.sigmoid(emb_mm_gate), emb_mm)
        emb_seq_gate = tf.matmul(emb_seq, self.weight_graph)
        emb_seq_gate = tf.multiply(tf.sigmoid(emb_seq_gate), emb_seq)
        ui = emb_oo_gate + emb_mm_gate + emb_seq_gate + emb_user
        return ui

    def _build_model(self):
        self._create_placeholder()
        self._init_constant()
        self._init_variable()
        user_emb = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # b*d
        item_embeddings_seq = self._forward_head_emb(self.item_embeddings_ii)
        item_embeddings_oo = tf.nn.embedding_lookup(self.item_embeddings, self.head_ph)
        item_embeddings_oo = self._One_one(item_embeddings_oo, self.item_embeddings_weight)
        item_embeddings_oo = tf.reduce_mean(item_embeddings_oo, axis=1, keepdims=True)
        item_embeddings_oo = tf.squeeze(item_embeddings_oo, axis=1)
        item_embeddings_mm = tf.nn.embedding_lookup(self.item_embeddings, self.max_ph)
        item_embeddings_mm = self.Multi_Multi(item_embeddings_mm)
        item_embeddings_mm = tf.reduce_mean(item_embeddings_mm, axis=1)

        pos_tail_emb = tf.nn.embedding_lookup(self.item_embeddings_ii, self.pos_tail_ph) 
        neg_tail_emb = tf.nn.embedding_lookup(self.item_embeddings_ii, self.neg_tail_ph) 

        pos_tail_bias = tf.gather(self.item_biases, self.pos_tail_ph)  
        neg_tail_bias = tf.gather(self.item_biases, self.neg_tail_ph) 

        pre_emb = self._fusion_gate(item_embeddings_oo, item_embeddings_mm, item_embeddings_seq, user_emb)
        pre_emb = tf.expand_dims(pre_emb, axis=1)
        pos_rating = -l2_distance(pre_emb, pos_tail_emb) + pos_tail_bias 
        neg_rating = -l2_distance(pre_emb, neg_tail_emb) + neg_tail_bias 


        pairwise_loss = tf.reduce_sum(log_loss(pos_rating - neg_rating))
        emb_reg = l2_loss(user_emb, item_embeddings_seq, item_embeddings_mm, item_embeddings_oo,
                          pos_tail_emb, neg_tail_emb, pos_tail_bias, neg_tail_bias)

        weight_reg = tf.reduce_sum(tf.square(self.item_embeddings_weight)) + \
                         tf.reduce_sum(tf.square(self.weight_oo)) + \
                         tf.reduce_sum(tf.square(self.weight_mm)) + \
                        tf.reduce_sum(tf.square(self.weight_graph)) + \
                     tf.reduce_sum(tf.square(self.weight_time))

            # objective loss and optimizer
        obj_loss = pairwise_loss + self.reg * emb_reg + weight_reg
        self.update_opt = tf.train.AdamOptimizer(self.lr).minimize(obj_loss)

        # for prediction
        self.item_embeddings_final = tf.Variable(tf.zeros([self.items_num, self.factors_num]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.assign_opt = tf.assign(self.item_embeddings_final, self.item_embeddings_ii[:self.items_num, :])
        j_emb = tf.expand_dims(self.item_embeddings_final, axis=0)
        self.prediction = -l2_distance(pre_emb, j_emb) + self.item_biases

    def train_model(self):
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data, len_seqs=self.n_max_length, len_next=self.n_next,
                                             pad=self.items_num, num_neg=self.n_next_neg,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=False)
        bets_result = 0.0
        counter = 0
        best_str = ""
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.global_step+1, self.epochs):
            for bat_users, bat_max, bat_pos_tail, bat_neg_tail in data_iter:
                bat_head = bat_max[:, -self.n_seqs:]
                bat_time = []
                for u in bat_users:
                    time_list = []
                    if len(self.user_pos_time[u])<self.n_seqs+1:
                        for i in range(self.n_seqs+1 - len(self.user_pos_time[u])):
                            time_list.insert(0, 0.001)
                        for i in range(len(self.user_pos_time[u])-1):
                            time_list.append((self.user_pos_time[u][i+1]-self.user_pos_time[u][i])/(3600))
                    else:
                        a = len(self.user_pos_time[u])
                        for i in range(self.n_seqs):
                            time_list.insert(0, (self.user_pos_time[u][a-i-1] - self.user_pos_time[u][a-i-2])/(3600))
                    bat_time.append(time_list)
                feed = {self.user_ph: bat_users,
                        self.max_ph: bat_max.reshape([-1, self.n_max_length]),
                        self.head_ph: bat_head.reshape([-1, self.n_seqs]),
                        self.pos_tail_ph: bat_pos_tail.reshape([-1, self.n_next]),
                        self.neg_tail_ph: bat_neg_tail.reshape([-1, self.n_next_neg]),
                        self.is_training: True,
                        self.timenow_ph: bat_time
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
            # save model
            if self.save_interval and epoch % self.save_interval == 0:
                self.save_model(global_step=epoch)

        self.logger.info("best:\t%s" % best_str)

    def evaluate_model(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, users):
        last_items = [self.test_item_seqs[u] for u in users]
        bat_time = []
        for u in users:
            time_list = []
            if len(self.user_pos_time[u]) < self.n_seqs + 1:
                for i in range(self.n_seqs + 1 - len(self.user_pos_time[u])):
                    time_list.insert(0, 0.001)
                for i in range(len(self.user_pos_time[u]) - 1):
                    time_list.append((self.user_pos_time[u][i + 1] - self.user_pos_time[u][i]) / (3600))
            else:
                a = len(self.user_pos_time[u])
                for i in range(self.n_seqs):
                    time_list.insert(0, (self.user_pos_time[u][a - i - 1] - self.user_pos_time[u][a - i - 2]) / (3600))
            bat_time.append(time_list)
        last_items_head = [last_items[i][-self.n_seqs:] for i in range(len(last_items))]
        feed = {self.user_ph: users, self.max_ph: last_items, self.head_ph: last_items_head, self.is_training: False, self.timenow_ph: bat_time}
        bat_ratings = self.sess.run(self.prediction, feed_dict=feed)
        return bat_ratings
