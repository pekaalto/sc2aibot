import numpy as np
import tensorflow as tf
from common.util import (
    weighted_random_sample, select_from_each_row, calculate_n_step_reward, combine_first_dimensions,
    ravel_index_pairs)


class TestUtil(tf.test.TestCase):
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
