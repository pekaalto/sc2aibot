import unittest

import numpy as np
import tensorflow as tf
from common.util import (
    weighted_random_sample, select_from_each_row, calculate_n_step_reward, combine_first_dimensions,
    ravel_index_pairs, general_n_step_advantage, dict_of_lists_to_list_of_dicst)


class TestUtil(tf.test.TestCase):
    def test_dict_list_transpose(self):
        x = {
            "a": [1, 2, 3],
            "b": [np.array([5, 6]), np.array([7, 8]), np.array([90, 100])]
        }
        result = dict_of_lists_to_list_of_dicst(x)
        expected = [
            {'a': 1, 'b': np.array([5, 6])},
            {'a': 2, 'b': np.array([7, 8])},
            {'a': 3, 'b': np.array([90, 100])}
        ]
        assert len(result) == len(expected)
        for r, e in zip(expected, result):
            assert r.keys() == e.keys()
            assert r["a"] == e["a"]
            self.assertAllEqual(r["b"], e["b"])

    def test_weighted_random_sample(self):

        probs = np.array([
            [1, 2, 4],
            [2, 4, 4],
            [0, 1, 2]
        ])

        p = weighted_random_sample(probs)
        n_sample = 5000
        with self.test_session() as sess:
            X = np.array([sess.run(p) for _ in range(n_sample)])

        # normalizing rows
        probs_norm = probs / probs.sum(axis=1, keepdims=True)
        expected_items_by_row = [
            [0, 1, 2],
            [0, 1, 2],
            [1, 2]
        ]
        for i in range(3):
            idx, counts = np.unique(X[:, i], return_counts=True)
            self.assertAllEqual(idx, expected_items_by_row[i])
            self.assertArrayNear(counts / n_sample, probs_norm[i][probs_norm[i] > 0], err=0.02)

    def test_select_from_each_row(self):
        x = np.random.rand(4, 5)
        col_idx = np.array([0, 0, 1, 2])
        with self.test_session():
            selection = select_from_each_row(x, col_idx).eval()

        assert selection.shape == (4,)

        for i, s in enumerate(selection):
            assert s == x[i, col_idx[i]]

    def test_calculate_n_step_reward(self):
        """
        compare_with_open_ai
        """

        def discount_with_dones(rewards, dones, gamma):
            """
            This is the openAI baselines implementation
            """
            discounted = []
            r = 0
            for reward, done in zip(rewards[::-1], dones[::-1]):
                r = reward + gamma * r * (1. - done)
                discounted.append(r)
            return discounted[::-1]

        rewards = np.arange(2, 6)
        gamma = 0.9
        last = 412.21
        rewards_with_last = np.append(rewards, last)

        res1 = calculate_n_step_reward(rewards.reshape(1, -1), gamma, np.array([last]))[0]
        res2 = discount_with_dones(
            rewards_with_last,
            np.zeros_like(rewards_with_last),
            gamma
        )[:-1]
        self.assertAllClose(res1, res2)

    def test_calculate_n_step_reward1(self):
        rewards = np.array([
            [1.0, 1.5],
            [0.0, 2.0]
        ])
        gamma = 0.9
        last = np.array([3.0, 1.0])

        res1 = calculate_n_step_reward(rewards, gamma, last)

        expected = np.array([
            [1.0 + 1.5 * 0.9 + 3 * 0.9 ** 2, 1.5 + 0.9 * 3],
            [2.0 * 0.9 + 1 * 0.9 ** 2, 2.0 + 0.9 * 1.0],
        ])

        self.assertAllClose(expected, res1)

    def test_combine_first_dimensions(self):
        x = np.random.randint(100, size=(3, 5, 7, 9))

        result = combine_first_dimensions(x)
        assert result.shape == (15, 7, 9)
        assert np.all(result[0] == x[0, 0])
        assert np.all(result[6] == x[1, 1])
        assert np.all(result[7] == x[1, 2])

        x = np.random.randint(10, size=(3, 5))
        assert np.all(combine_first_dimensions(x) == x.flatten())

    def test_ravel_index_pairs(self):
        pairs = np.array([
            [1, 6],
            [3, 3],
            [5, 2]
        ])

        for x in [
            np.random.randint(1000, size=(3, 10, 8)),
            np.random.randint(1000, size=(3, 12, 8)),
            np.random.randint(1000, size=(3, 8, 8))
        ]:
            expected_slice = x[np.arange(3), pairs[:, 0], pairs[:, 1]]

            with self.test_session():
                flat_idx = ravel_index_pairs(pairs, n_col=8).eval()
                wrong_idx = ravel_index_pairs(pairs, n_col=10).eval()

            self.assertAllEqual(x.reshape(3, -1)[np.arange(3), flat_idx], expected_slice)
            self.assertAllEqual(flat_idx, [14, 27, 42])
            assert (wrong_idx != flat_idx).sum() > 0


class TestGeneralNStepAdvantage(tf.test.TestCase):
    def test_gamma_one(self):
        """
        compare to simpler calculation when lambda = 1
        """

        value_estimates = np.random.rand(5, 11)
        one_step_rewards = np.random.rand(5, 10)
        discount = 0.76

        a1 = general_n_step_advantage(
            one_step_rewards,
            value_estimates,
            discount=discount,
            lambda_par=1.0
        )

        a2 = calculate_n_step_reward(
            one_step_rewards,
            discount=discount,
            last_state_values=value_estimates[:, -1]
        ) - value_estimates[:, :-1]

        self.assertAllClose(a1, a2)

    def test_gamma_zero(self):
        value_estimates = np.random.rand(5, 11) * 10
        one_step_rewards = np.random.rand(5, 10) * 10
        discount = 0.76

        a1 = general_n_step_advantage(
            one_step_rewards,
            value_estimates,
            discount=discount,
            lambda_par=1e-7
        )

        a2 = general_n_step_advantage(
            one_step_rewards,
            value_estimates,
            discount=discount,
            lambda_par=0.0
        )

        a3 = one_step_rewards + discount * value_estimates[:, 1:] - value_estimates[:, :-1]

        self.assertAllClose(a1, a3, rtol=1e-4, atol=1e-4)
        self.assertAllClose(a2, a3)

    def test_general(self):
        value_estimates = np.array([1, 1.5, 0.5, 2.0])
        one_step_rewards = np.array([-1, 2.0, 5.0])
        discount = 0.75
        lambda_par = 0.5
        # delta_t = r_t + gamma*V_{t+1} - V_{t}
        # A_t = delta_t + (gamma * lambda)**1 delta_{t+1} + ... + (gamma * lambda)**(T - t + 1) delta_{T - 1}
        # T = len(one_step_rewards) + 1
        deltas = np.zeros(3)
        for i in range(3):
            deltas[i] = one_step_rewards[i] + discount * value_estimates[i + 1] - value_estimates[i]

        expected_advantages = np.zeros(3)
        expected_advantages[0] = (
            deltas[0] +
            (discount * lambda_par) * deltas[1] +
            (discount * lambda_par) ** 2 * deltas[2]
        )
        expected_advantages[1] = (
            deltas[1] +
            (discount * lambda_par) * deltas[2]
        )
        expected_advantages[2] = deltas[2]

        a = general_n_step_advantage(
            one_step_rewards[np.newaxis, ...],
            value_estimates[np.newaxis, ...],
            discount=discount,
            lambda_par=lambda_par
        )
        assert a.shape == (1, 3)
        self.assertAllClose(expected_advantages, a[0])

    def test_general_batch(self):
        value_estimates = np.random.rand(3, 11) * 10
        one_step_rewards = np.random.rand(3, 10) * 10

        discount = 0.84
        lambda_par = 0.45

        a1 = general_n_step_advantage(
            one_step_rewards,
            value_estimates,
            discount=discount,
            lambda_par=lambda_par
        )

        a2 = general_n_step_advantage(
            one_step_rewards[[1]],
            value_estimates[[1]],
            discount=discount,
            lambda_par=lambda_par
        )

        self.assertAllClose(a1[[1]], a2)

    def test_general_n_step_advantage_3(self):
        # 1 dimensional input
        with self.assertRaises(Exception):
            general_n_step_advantage(
                np.random.rand(10),
                np.random.rand(11),
                discount=1.0,
                lambda_par=1.0
            )

        # wrong input dim
        with self.assertRaises(Exception):
            general_n_step_advantage(
                np.random.rand(2, 5),
                np.random.rand(2, 5),
                discount=1.0,
                lambda_par=1.0
            )
