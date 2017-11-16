import collections
import os

import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers.optimizers import OPTIMIZER_SUMMARIES
from common.preprocess import ObsProcesser, FEATURE_KEYS, AgentInputTuple
from common.util import weighted_random_sample, select_from_each_row, ravel_index_pairs


def _get_placeholders(spatial_dim):
    sd = spatial_dim
    feature_list = [
        (FEATURE_KEYS.minimap_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_MINIMAP_CHANNELS]),
        (FEATURE_KEYS.screen_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_SCREEN_CHANNELS]),
        (FEATURE_KEYS.screen_unit_type, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.is_spatial_action_available, tf.float32, [None]),
        (FEATURE_KEYS.available_action_ids, tf.float32, [None, len(actions.FUNCTIONS)]),
        (FEATURE_KEYS.selected_spatial_action, tf.int32, [None, 2]),
        (FEATURE_KEYS.selected_action_id, tf.int32, [None]),
        (FEATURE_KEYS.value_target, tf.float32, [None]),
        (FEATURE_KEYS.player_relative_screen, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.player_relative_minimap, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.advantage, tf.float32, [None])
    ]
    return AgentInputTuple(
        **{name: tf.placeholder(dtype, shape, name) for name, dtype, shape in feature_list}
    )


class ACMode:
    A2C = "a2c"
    PPO = "ppo"


SelectedLogProbs = collections.namedtuple("SelectedLogProbs", ["action_id", "spatial", "total"])


class ActorCriticAgent:
    _scalar_summary_key = "scalar_summaries"

    def __init__(self,
            sess,
            summary_path,
            all_summary_freq,
            scalar_summary_freq,
            spatial_dim,
            mode,
            clip_epsilon=0.2,
            unit_type_emb_dim=4,
            loss_value_weight=1.0,
            entropy_weight_spatial=1e-6,
            entropy_weight_action_id=1e-5,
            max_gradient_norm=None,
            optimiser="adam",
            optimiser_pars=None
    ):
        """
        Agent to for learning pysc2-minigames using
        -a2c: synchronous version https://blog.openai.com/baselines-acktr-a2c/
            of the original a3c algorithm https://arxiv.org/pdf/1602.01783.pdf
        - FullyConvPolicy from https://deepmind.com/documents/110/sc2le.pdf

        Other policies can be specified by overriding or using other function in place of
        _build_fullyconv_network

        some ideas here are borrowed from
        https://github.com/xhujoy/pysc2-agents
        but this is still a different implementation

        :param str summary_path: tensorflow summaries will be created here
        :param int all_summary_freq: how often save all summaries
        :param int scalar_summary_freq: int, how often save scalar summaries
        :param int spatial_dim: dimension for both minimap and screen
        :param float loss_value_weight: value weight for a2c update
        :param float entropy_weight_spatial: spatial entropy weight for a2c update
        :param float entropy_weight_action_id: action selection entropy weight for a2c update
        :param str optimiser: see valid choiches below
        :param dict optimiser_pars: optional parameters to pass in optimiser
        """

        assert optimiser in ["adam", "rmsprop"]
        assert mode in [ACMode.A2C, ACMode.PPO]
        self.mode = mode
        self.sess = sess
        self.spatial_dim = spatial_dim
        self.loss_value_weight = loss_value_weight
        self.entropy_weight_spatial = entropy_weight_spatial
        self.entropy_weight_action_id = entropy_weight_action_id
        self.unit_type_emb_dim = unit_type_emb_dim
        self.summary_path = summary_path
        os.makedirs(summary_path, exist_ok=True)
        self.summary_writer = tf.summary.FileWriter(summary_path)
        self.all_summary_freq = all_summary_freq
        self.scalar_summary_freq = scalar_summary_freq
        self.train_step = 0
        self.max_gradient_norm = max_gradient_norm
        self.clip_epsilon = clip_epsilon

        opt_class = tf.train.AdamOptimizer if optimiser == "adam" else tf.train.RMSPropOptimizer
        if optimiser_pars is None:
            pars = {
                "adam": {
                    "learning_rate": 1e-4,
                    "epsilon": 5e-7
                },
                "rmsprop": {
                    "learning_rate": 2e-4
                }
            }[optimiser]
        else:
            pars = optimiser_pars
        self.optimiser = opt_class(**pars)

    def init(self):
        self.sess.run(self.init_op)
        if self.mode == ACMode.PPO:
            self.update_theta()

    def _get_select_action_probs(self, pi, selected_spatial_action_flat):
        action_id = select_from_each_row(
            pi.action_id_log_probs, self.placeholders.selected_action_id
        )
        spatial = select_from_each_row(
            pi.spatial_action_log_probs, selected_spatial_action_flat
        )
        total = spatial + action_id

        return SelectedLogProbs(action_id, spatial, total)

    def _scalar_summary(self, name, tensor):
        tf.summary.scalar(name, tensor,
            collections=[tf.GraphKeys.SUMMARIES, self._scalar_summary_key])

    def build_model(self):
        self.placeholders = _get_placeholders(self.spatial_dim)

        with tf.variable_scope("theta"):
            theta = FullyConvPolicy(self, trainable=True).build()

        selected_spatial_action_flat = ravel_index_pairs(
            self.placeholders.selected_spatial_action, self.spatial_dim
        )

        selected_log_probs = self._get_select_action_probs(theta, selected_spatial_action_flat)

        # maximum is to avoid 0 / 0 because this is used to calculate some means
        sum_spatial_action_available = tf.maximum(
            1e-10, tf.reduce_sum(self.placeholders.is_spatial_action_available)
        )

        neg_entropy_spatial = tf.reduce_sum(
            theta.spatial_action_probs * theta.spatial_action_log_probs
        ) / sum_spatial_action_available
        neg_entropy_action_id = tf.reduce_mean(tf.reduce_sum(
            theta.action_id_probs * theta.action_id_log_probs, axis=1
        ))

        if self.mode == ACMode.PPO:
            # could also use stop_gradient and forget about the trainable
            with tf.variable_scope("theta_old"):
                theta_old = FullyConvPolicy(self, trainable=False).build()

            new_theta_var = tf.global_variables("theta/")
            old_theta_var = tf.global_variables("theta_old/")

            assert len(tf.trainable_variables("theta/")) == len(new_theta_var)
            assert not tf.trainable_variables("theta_old/")
            assert len(old_theta_var) == len(new_theta_var)

            self.update_theta_op = [
                tf.assign(t_old, t_new) for t_new, t_old in zip(new_theta_var, old_theta_var)
            ]

            selected_log_probs_old = self._get_select_action_probs(
                theta_old, selected_spatial_action_flat
            )
            ratio = tf.exp(selected_log_probs.total - selected_log_probs_old.total)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
            )
            l_clip = tf.minimum(
                ratio * self.placeholders.advantage,
                clipped_ratio * self.placeholders.advantage
            )
            self.sampled_action_id = weighted_random_sample(theta_old.action_id_probs)
            self.sampled_spatial_action = weighted_random_sample(theta_old.spatial_action_probs)
            self.value_estimate = theta_old.value_estimate
            self._scalar_summary("action/ratio", tf.reduce_mean(clipped_ratio))
            self._scalar_summary("action/ratio_is_clipped",
                tf.reduce_mean(tf.to_float(tf.equal(ratio, clipped_ratio))))
            policy_loss = -tf.reduce_mean(l_clip)
        else:
            self.sampled_action_id = weighted_random_sample(theta.action_id_probs)
            self.sampled_spatial_action = weighted_random_sample(theta.spatial_action_probs)
            self.value_estimate = theta.value_estimate
            policy_loss = -tf.reduce_mean(selected_log_probs.total * self.placeholders.advantage)

        value_loss = tf.losses.mean_squared_error(
            self.placeholders.value_target, theta.value_estimate)

        loss = (
            policy_loss
            + value_loss * self.loss_value_weight
            + neg_entropy_spatial * self.entropy_weight_spatial
            + neg_entropy_action_id * self.entropy_weight_action_id
        )

        self.train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=self.optimiser,
            clip_gradients=self.max_gradient_norm,
            summaries=OPTIMIZER_SUMMARIES,
            learning_rate=None,
            name="train_op"
        )

        self._scalar_summary("value/estimate", tf.reduce_mean(self.value_estimate))
        self._scalar_summary("value/target", tf.reduce_mean(self.placeholders.value_target))
        self._scalar_summary("action/is_spatial_action_available",
            tf.reduce_mean(self.placeholders.is_spatial_action_available))
        self._scalar_summary("action/selected_id_log_prob",
            tf.reduce_mean(selected_log_probs.action_id))
        self._scalar_summary("loss/policy", policy_loss)
        self._scalar_summary("loss/value", value_loss)
        self._scalar_summary("loss/neg_entropy_spatial", neg_entropy_spatial)
        self._scalar_summary("loss/neg_entropy_action_id", neg_entropy_action_id)
        self._scalar_summary("loss/total", loss)
        self._scalar_summary("value/advantage", tf.reduce_mean(self.placeholders.advantage))
        self._scalar_summary("action/selected_total_log_prob",
            tf.reduce_mean(selected_log_probs.total))
        self._scalar_summary("action/selected_spatial_log_prob",
            tf.reduce_sum(selected_log_probs.spatial) / sum_spatial_action_available)

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.all_summary_op = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)
        self.scalar_summary_op = tf.summary.merge(tf.get_collection(self._scalar_summary_key))

    def _input_to_feed_dict(self, input_dict):
        return {k + ":0": v for k, v in input_dict.items()}

    def step(self, obs):
        feed_dict = self._input_to_feed_dict(obs)

        action_id, spatial_action, value_estimate = self.sess.run(
            [self.sampled_action_id, self.sampled_spatial_action, self.value_estimate],
            feed_dict=feed_dict
        )

        spatial_action_2d = np.array(
            np.unravel_index(spatial_action, (self.spatial_dim,) * 2)
        ).transpose()

        return action_id, spatial_action_2d, value_estimate

    def train(self, input_dict):
        feed_dict = self._input_to_feed_dict(input_dict)
        ops = [self.train_op]

        write_all_summaries = (
            (self.train_step % self.all_summary_freq == 0) and
            self.summary_path is not None
        )
        write_scalar_summaries = (
            (self.train_step % self.scalar_summary_freq == 0) and
            self.summary_path is not None
        )

        if write_all_summaries:
            ops.append(self.all_summary_op)
        elif write_scalar_summaries:
            ops.append(self.scalar_summary_op)

        r = self.sess.run(ops, feed_dict)

        if write_all_summaries or write_scalar_summaries:
            self.summary_writer.add_summary(r[-1], global_step=self.train_step)


        self.train_step += 1

    def get_value(self, obs):
        feed_dict = self._input_to_feed_dict(obs)
        return self.sess.run(self.value_estimate, feed_dict=feed_dict)

    def flush_summaries(self):
        self.summary_writer.flush()

    def save(self, path, step=None):
        os.makedirs(path, exist_ok=True)
        step = step or self.train_step
        print("saving model to %s, step %d" % (path, step))
        self.saver.save(self.sess, path + '/model.ckpt', global_step=step)

    def load(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("loaded old model with train_step %d" % self.train_step)
        self.train_step += 1

    def update_theta(self):
        if self.mode == ACMode.PPO:
            self.sess.run(self.update_theta_op)


class FullyConvPolicy:
    def __init__(self,
            agent: ActorCriticAgent,
            trainable: bool = True
    ):
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=16,
            kernel_size=5,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=3,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)

        return conv2

    def build(self):
        units_embedded = layers.embed_sequence(
            self.placeholders.screen_unit_type,
            vocab_size=SCREEN_FEATURES.unit_type.scale,
            embed_dim=self.unittype_emb_dim,
            scope="unit_type_emb",
            trainable=self.trainable
        )

        # Let's not one-hot zero which is background
        player_relative_screen_one_hot = layers.one_hot_encoding(
            self.placeholders.player_relative_screen,
            num_classes=SCREEN_FEATURES.player_relative.scale
        )[:, :, :, 1:]
        player_relative_minimap_one_hot = layers.one_hot_encoding(
            self.placeholders.player_relative_minimap,
            num_classes=MINIMAP_FEATURES.player_relative.scale
        )[:, :, :, 1:]

        channel_axis = 3
        screen_numeric_all = tf.concat(
            [self.placeholders.screen_numeric, units_embedded, player_relative_screen_one_hot],
            axis=channel_axis
        )
        minimap_numeric_all = tf.concat(
            [self.placeholders.minimap_numeric, player_relative_minimap_one_hot],
            axis=channel_axis
        )
        screen_output = self._build_convs(screen_numeric_all, "screen_network")
        minimap_output = self._build_convs(minimap_numeric_all, "minimap_network")

        map_output = tf.concat([screen_output, minimap_output], axis=channel_axis)

        spatial_action_logits = layers.conv2d(
            map_output,
            data_format="NHWC",
            num_outputs=1,
            kernel_size=1,
            stride=1,
            activation_fn=None,
            scope='spatial_action',
            trainable=self.trainable
        )

        spatial_action_probs = tf.nn.softmax(layers.flatten(spatial_action_logits))

        map_output_flat = layers.flatten(map_output)

        fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        action_id_probs = layers.fully_connected(
            fc1,
            num_outputs=len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1
        action_id_probs *= self.placeholders.available_action_ids
        action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keep_dims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        spatial_action_log_probs = (
            logclip(spatial_action_probs)
            * tf.expand_dims(self.placeholders.is_spatial_action_available, axis=1)
        )

        # non-available actions get log(1e-10) value but that's ok because it's never used
        action_id_log_probs = logclip(action_id_probs)

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        self.spatial_action_probs = spatial_action_probs
        self.action_id_log_probs = action_id_log_probs
        self.spatial_action_log_probs = spatial_action_log_probs
        return self
