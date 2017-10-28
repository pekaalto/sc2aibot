from unittest import TestCase
from common.preprocess import ActionProcesser, ObsProcesser
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib import actions
from pysc2.lib.actions import FunctionCall
import numpy as np
from pysc2.lib.features import Features, SCREEN_FEATURES, FeatureType, MINIMAP_FEATURES


class TestActionProcesser(TestCase):
    def test_simple(self):
        a = ActionProcesser(
            dim=40,
            rect_delta=5,
        )

        action_ids = [2, 331, 0, 1]
        coords = ((15, 4), (22, 33), (1, 1), (1, 1))

        actions = a.process(
            action_ids,
            coords
        )

        expected = [
            FunctionCall(function=2, arguments=[[0], (4, 15)]),
            FunctionCall(function=331, arguments=[[0], (33, 22)]),
            FunctionCall(function=0, arguments=[]),
            FunctionCall(function=1, arguments=[(1, 1)])
        ]

        assert actions == expected

    def test_rectangle(self):
        dim = 48
        a = ActionProcesser(
            dim=dim,
            rect_delta=7,
        )

        action_ids = [3, 3, 3, 3]
        coords = ((15, 4), (22, 33), (1, 1), (45, 10))

        actions = a.process(
            action_ids,
            coords
        )

        expected = [
            FunctionCall(function=3, arguments=[[0], [8, 0], [22, 11]]),
            FunctionCall(function=3, arguments=[[0], [15, 26], [29, 40]]),
            FunctionCall(function=3, arguments=[[0], [0, 0], [8, 8]]),
            FunctionCall(function=3, arguments=[[0], [38, 3], [47, 17]])
        ]

        assert actions == expected

    def test_invalid_and_dim(self):
        action = ([3], [[14, 3]])
        with self.assertRaises(AssertionError):
            ActionProcesser(dim=5, rect_delta=7).process(*action)

        assert ActionProcesser(dim=15, rect_delta=2).process(*action) == \
               [FunctionCall(function=3, arguments=[[0], [12, 1], [14, 5]])]

        assert ActionProcesser(dim=40, rect_delta=2).process(*action) == \
               [FunctionCall(function=3, arguments=[[0], [12, 1], [16, 5]])]


class TestObsProcesser:
    """
    This is more of syntax check and notes. Nothing very exact is tested here
    """

    def test_one_input(self):
        d = 48

        # These shapes are what is actually returned from environment
        dummy_obs = {
            "screen": np.zeros((16, d, d), dtype="int32"),
            "minimap": np.zeros((7, d, d), dtype="int32"),
            "available_actions": np.arange(10)
        }

        dummy_ts = TimeStep(StepType.MID, 0.0, 0.0, dummy_obs)

        p = ObsProcesser()

        assert p.process_one_input(dummy_ts)["screen_numeric"].shape == (
            ObsProcesser.N_SCREEN_CHANNELS, d, d)
        assert p.process_one_input(dummy_ts)["minimap_numeric"].shape == (
            ObsProcesser.N_MINIMAP_CHANNELS, d, d)

        n_screen_scalar_features = len(
            [k for k in SCREEN_FEATURES if k.type == FeatureType.SCALAR])

        total_screen_dim = n_screen_scalar_features + 3 + 1  # binary flags + visibility_flag
        assert total_screen_dim == ObsProcesser.N_SCREEN_CHANNELS

        n_screen_minimap_features = len(
            [k for k in MINIMAP_FEATURES if k.type == FeatureType.SCALAR])
        total_minimap_dim = n_screen_minimap_features + 3 + 1
        assert total_minimap_dim == ObsProcesser.N_MINIMAP_CHANNELS
