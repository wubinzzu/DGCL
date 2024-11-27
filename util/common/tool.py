__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["Reduction", "InitArg", "normalize_adj_matrix"]

import numpy as np
import scipy.sparse as sp
from reckit import randint_choice
from functools import wraps
import time



class Reduction(object):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"

    @classmethod
    def all(cls):
        return (cls.NONE,
                cls.SUM,
                cls.MEAN)

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            key_list = ', '.join(cls.all())
            raise ValueError(f"{key} is an invalid Reduction Key, which must be one of '{key_list}'.")


class InitArg(object):
    MEAN = 0.0
    STDDEV = 0.01
    MIN_VAL = -0.05
    MAX_VAL = 0.05


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


def pad_sequences(sequences, value=0., max_len=None,
                  padding='post', truncating='post', dtype=np.int32):
    """Pads sequences to the same length.

    Args:
        sequences (list): A list of lists, where each element is a sequence.
        value (int or float): Padding value. Defaults to `0.`.
        max_len (int or None): Maximum length of all sequences.
        padding (str): `"pre"` or `"post"`: pad either before or after each
            sequence. Defaults to `post`.
        truncating (str): `"pre"` or `"post"`: remove values from sequences
            larger than `max_len`, either at the beginning or at the end of
            the sequences. Defaults to `post`.
        dtype (int or float): Type of the output sequences. Defaults to `np.int32`.

    Returns:
        np.ndarray: Numpy array with shape `(len(sequences), max_len)`.

    Raises:
        ValueError: If `padding` or `truncating` is not understood.
    """
    if max_len is None:
        max_len = np.max([len(x) for x in sequences])

    x = np.full([len(sequences), max_len], value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def batch_randint_choice(high, size, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).

    Args:
        high (int):
        size: 1-D array_like
        replace (bool):
        p: 2-D array_like
        exclusion: a list of 1-D array_like

    Returns:
        list: a list of 1-D array_like sample

    """
    if p is not None:
        raise NotImplementedError

    if exclusion is not None and len(size) != len(exclusion):
        raise ValueError("The shape of 'exclusion' is not compatible with the shape of 'size'!")

    results = []
    for idx in range(len(size)):
        p_tmp = p[idx] if p is not None else None
        exc = exclusion[idx] if exclusion is not None else None
        results.append(randint_choice(high, size=size[idx], replace=replace, p=p_tmp, exclusion=exc))
    return results

def csr_to_user_dict(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        if len(value.indices) >= 0:
            train_dict[idx] = value.indices.copy().tolist()
    return train_dict

def csr_to_user_dict_bytime(time_matrix,train_matrix):
    train_dict = {}
    time_matrix = time_matrix
    user_pos_items = csr_to_user_dict(train_matrix)
    for u, items in user_pos_items.items():
        sorted_items = sorted(items, key=lambda x: time_matrix[u,x])
        train_dict[u] = np.array(sorted_items, dtype=np.int32).tolist()
    a = train_dict
    return train_dict

def csr_to_time_dict(time_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    time_dict = {}
    for idx, value in enumerate(time_matrix):
        if len(value.indices) > 0:
            time_dict[idx] = value.data.copy().tolist()
    return time_dict

def timer(func):
    """The timer decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result
    return wrapper



