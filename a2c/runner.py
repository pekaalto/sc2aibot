from collections import namedtuple

import numpy as np
import sys
from a2c.agent import ActorCriticAgent, ACMode
from common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from common.util import calculate_n_step_reward, general_n_step_advantage, combine_first_dimensions, \
    dict_of_lists_to_list_of_dicst
import tensorflow as tf
from absl import flags

PPORunParams = namedtuple("PPORunParams", ["lambda_par", "batch_size", "n_epochs"])


class Runner(object):
    def __init__(
            self,
            envs,
            agent: ActorCriticAgent,
            n_steps=5,
            discount=0.99,
            do_training=True,
            ppo_par: PPORunParams = None
    ):
        self.envs = envs
        self.agent = agent
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.n_steps = n_steps
        self.discount = discount
        self.do_training = do_training
        self.ppo_par = ppo_par
        self.batch_counter = 0
        self.episode_counter = 0
        assert self.agent.mode in [ACMode.PPO, ACMode.A2C]
        self.is_ppo = self.agent.mode == ACMode.PPO
        if self.is_ppo:
            assert ppo_par is not None
            assert n_steps * envs.n_envs % ppo_par.batch_size == 0
            assert n_steps * envs.n_envs >= ppo_par.batch_size
            self.ppo_par = ppo_par

    def reset(self):
        obs = self.envs.reset()
        self.latest_obs = self.obs_processer.process(obs)

    def _log_score_to_tb(self, score):
        summary = tf.Summary()
        summary.value.add(tag='sc2/episode_score', simple_value=score)
        self.agent.summary_writer.add_summary(summary, self.episode_counter)

    def _handle_episode_end(self, timestep):
        score = timestep.observation["score_cumulative"][0]
        print("episode %d ended. Score %f" % (self.episode_counter, score))
        self._log_score_to_tb(score)
        self.episode_counter += 1

    def _train_ppo_epoch(self, full_input):
        total_obs = self.n_steps * self.envs.n_envs
        shuffle_idx = np.random.permutation(total_obs)
        batches = dict_of_lists_to_list_of_dicst({
            k: np.split(v[shuffle_idx], total_obs // self.ppo_par.batch_size)
            for k, v in full_input.items()
        })
        for b in batches:
            self.agent.train(b)

    def run_batch(self):
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.envs.n_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.n_envs, self.n_steps), dtype=np.float32)

        latest_obs = self.latest_obs

        for n in range(self.n_steps):
            # could calculate value estimate from obs when do training
            # but saving values here will make n step reward calculation a bit easier
            action_ids, spatial_action_2ds, value_estimate = self.agent.step(latest_obs)

            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids, spatial_action_2ds))

            actions_pp = self.action_processer.process(action_ids, spatial_action_2ds)
            obs_raw = self.envs.step(actions_pp)
            latest_obs = self.obs_processer.process(obs_raw)
            mb_rewards[:, n] = [t.reward for t in obs_raw]

            for t in obs_raw:
                if t.last():
                    self._handle_episode_end(t)

        mb_values[:, -1] = self.agent.get_value(latest_obs)

        n_step_advantage = general_n_step_advantage(
            mb_rewards,
            mb_values,
            self.discount,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0
        )

        full_input = {
            # these are transposed because action/obs
            # processers return [time, env, ...] shaped arrays
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose()
        }
        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()}

        if not self.do_training:
            pass
        elif self.agent.mode == ACMode.A2C:
            self.agent.train(full_input)
        elif self.agent.mode == ACMode.PPO:
            for epoch in range(self.ppo_par.n_epochs):
                self._train_ppo_epoch(full_input)
            self.agent.update_theta()

        self.latest_obs = latest_obs
        self.batch_counter += 1
        sys.stdout.flush()
