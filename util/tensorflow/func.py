__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["inner_product", "euclidean_distance", "l2_distance",
           "get_initializer", "get_session",
           "sp_mat_to_sp_tensor", "dropout_sparse"]

import tensorflow as tf
from reckit import typeassert
from collections import OrderedDict
from util.common import InitArg
import numpy as np
import scipy.sparse as sp


_initializers = OrderedDict()
_initializers["normal"] = tf.initializers.random_normal(mean=InitArg.MEAN, stddev=InitArg.STDDEV)
_initializers["truncated_normal"] = tf.initializers.truncated_normal(mean=InitArg.MEAN, stddev=InitArg.STDDEV)
_initializers["uniform"] = tf.initializers.random_uniform(minval=InitArg.MIN_VAL, maxval=InitArg.MAX_VAL)
_initializers["he_normal"] = tf.initializers.he_normal()
_initializers["he_uniform"] = tf.initializers.he_uniform()
_initializers["xavier_normal"] = tf.initializers.glorot_normal()
_initializers["xavier_uniform"] = tf.initializers.glorot_uniform()
_initializers["zeros"] = tf.initializers.zeros()
_initializers["ones"] = tf.initializers.ones()


@typeassert(init_method=str)
def get_initializer(init_method):
    if init_method not in _initializers:
        init_list = ', '.join(_initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")
    return _initializers[init_method]


def inner_product(a, b):
    return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def euclidean_distance(a, b):
    return tf.norm(a - b, ord='euclidean', axis=-1)


l2_distance = euclidean_distance


def get_session(gpu_memory_fraction=None):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    if gpu_memory_fraction is not None:
        tf_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    return sess


def sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = np.asarray([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def dropout_sparse(tf_sp_mat, keep_prob, nnz):
    """Dropout for sparse tensors.
    """
    noise_shape = [nnz]
    random_tensor = tf.random_uniform(noise_shape) + keep_prob
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(tf_sp_mat, dropout_mask)
    scale = 1.0 / keep_prob
    return pre_out * scale

def normalize_adj_matrix(sp_mat, norm_method="left"):
    """Normalize adjacent matrix

    Args:
        sp_mat: A sparse adjacent matrix
        norm_method (str): The normalization method, can be 'symmetric'
            or 'left'.

    Returns:
        sp.spmatrix: The normalized adjacent matrix.

    """

    d_in = np.asarray(sp_mat.sum(axis=1))  # indegree
    if norm_method == "left":
        rec_d_in = np.power(d_in, -1).flatten()  # reciprocal
        rec_d_in[np.isinf(rec_d_in)] = 0.  # replace inf
        rec_d_in = sp.diags(rec_d_in)  # to diagonal matrix
        norm_sp_mat = rec_d_in.dot(sp_mat)  # left matmul
    elif norm_method == "symmetric":
        rec_sqrt_d_in = np.power(d_in, -0.5).flatten()
        rec_sqrt_d_in[np.isinf(rec_sqrt_d_in)] = 0.
        rec_sqrt_d_in = sp.diags(rec_sqrt_d_in)

        mid_sp_mat = rec_sqrt_d_in.dot(sp_mat)  # left matmul
        norm_sp_mat = mid_sp_mat.dot(rec_sqrt_d_in)  # right matmul
    else:
        raise ValueError(f"'{norm_method}' is an invalid normalization method.")

    return norm_sp_mat