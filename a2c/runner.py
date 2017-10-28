import numpy as np
import sys
from a2c.agent import A2CAgent
from common.preprocess import ObsProcesser, ActionProcesser
from common.util import calculate_n_step_reward
import tensorflow as tf
from absl import flags


class Runner(object):
    def __init__(
            self,
            envs,
            agent: A2CAgent,
            n_steps=5,
            discount=0.99,
            do_training=True
    ):
        self.envs = envs
        self.agent = agent
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.n_steps = n_steps
        self.discount = discount
        self.do_training = do_training
        self.batch_counter = 0
        self.episode_counter = 0

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

    def run_batch(self):
        dim = (self.envs.n_envs, self.n_steps)
        mb_rewards = np.zeros(dim, dtype=np.float32)
        mb_actions = []
        mb_obs = []

        latest_obs = self.latest_obs

        for n in range(self.n_steps):
            # values is not used for anything but calculated anyway
            action_ids, spatial_action_2ds = self.agent.step(latest_obs)

            mb_obs.append(latest_obs)
            mb_actions.append((action_ids, spatial_action_2ds))

            actions_pp = self.action_processer.process(action_ids, spatial_action_2ds)
            obs_raw = self.envs.step(actions_pp)
            latest_obs = self.obs_processer.process(obs_raw)
            mb_rewards[:, n] = [t.reward for t in obs_raw]

            for t in obs_raw:
                if t.last():
                    self._handle_episode_end(t)

        last_state_values = self.agent.get_value(latest_obs)

        n_step_rewards = calculate_n_step_reward(mb_rewards, self.discount, last_state_values)

        mb_actions_combined = self.action_processer.combine_batch(mb_actions)
        mb_obs_combined = self.obs_processer.combine_batch(mb_obs)

        if self.do_training:
            self.agent.train(
                n_step_rewards.transpose(),
                mb_obs_combined=mb_obs_combined,
                mb_actions_combined=mb_actions_combined
            )

        self.latest_obs = latest_obs
        self.batch_counter += 1
        sys.stdout.flush()
