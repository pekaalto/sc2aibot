import numpy as np
import tensorflow as tf


def weighted_random_sample(weights):
    """
    :param weights: 2d tensor [n, d] containing positive weights for sampling
    :return: 1d tensor [n] with idx in [0, d) randomly sampled proportional to weights
    """
    u = tf.random_uniform(tf.shape(weights))
    return tf.argmax(tf.log(u) / weights, axis=1)


def select_from_each_row(params, indices):
    """
    :param params: 2d tensor of shape [d1,d2]
    :param indices: 1d tensor of shape [d1] with values in [d1, d2)
    :return: 1d tensor of shape [d1] which has one value from each row of params selected with indices
    """
    sel = tf.stack([tf.range(tf.shape(params)[0]), indices], axis=1)
    return tf.gather_nd(params, sel)


def calculate_n_step_reward(
        one_step_rewards: np.ndarray,
        discount: float,
        last_state_values: np.ndarray):
    """
    :param one_step_rewards: [n_env, n_batch]
    :param discount: scalar discount paramater
    :param last_state_values: [n_env], bootstrap from these if not done
    :return:
    """

    discount = discount ** np.arange(one_step_rewards.shape[1], -1, -1)
    reverse_rewards = np.c_[one_step_rewards, last_state_values][:, ::-1]
    full_discounted_reverse_rewards = reverse_rewards * discount
    return (np.cumsum(full_discounted_reverse_rewards, axis=1) / discount)[:, :0:-1]


def combine_first_dimensions(x: np.ndarray):
    """
    :param x: array of [batch_size, time, ...]
    :returns array of [batch_size * time, ...]
    """
    first_dim = x.shape[0] * x.shape[1]
    other_dims = x.shape[2:]
    dims = (first_dim,) + other_dims
    return x.reshape(*dims)


def ravel_index_pairs(idx_pairs, n_col):
    return tf.reduce_sum(idx_pairs * np.array([n_col, 1])[np.newaxis, ...], axis=1)
