import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES
from tensorflow.contrib import framework
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers.optimizers import OPTIMIZER_SUMMARIES
from common.preprocess import ObsProcesser
from common.util import weighted_random_sample, select_from_each_row, combine_first_dimensions, \
    ravel_index_pairs


def _build_convs(inputs, name):
    """
    helper method for building screen and minimap conv networks
    """
    conv1 = layers.conv2d(
        inputs=inputs,
        data_format="NHWC",
        num_outputs=16,
        kernel_size=5,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        scope="%s/conv1" % name
    )
    conv2 = layers.conv2d(
        inputs=conv1,
        data_format="NHWC",
        num_outputs=32,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        scope="%s/conv2" % name
    )

    layers.summarize_activation(conv1)
    layers.summarize_activation(conv2)

    return conv2


class A2CAgent:
    def __init__(self,
            sess,
            summary_path,
            all_summary_freq,
            scalar_summary_freq,
            spatial_dim,
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

    def reset(self):
        pass

    def _define_input_placeholders(self):
        self.ph_minimap_numeric = tf.placeholder(tf.float32,
            [None, self.spatial_dim, self.spatial_dim, ObsProcesser.N_MINIMAP_CHANNELS],
            name='minimap_numeric')
        self.ph_screen_numeric = tf.placeholder(tf.float32,
            [None, self.spatial_dim, self.spatial_dim, ObsProcesser.N_SCREEN_CHANNELS],
            name='screen_numeric')
        self.ph_screen_unit_type = tf.placeholder(tf.int32,
            [None, self.spatial_dim, self.spatial_dim],
            name="screen_unit_type"
        )
        self.ph_is_spatial_action_available = tf.placeholder(tf.float32, [None],
            name='is_spatial_action_available')
        self.ph_available_action_ids = tf.placeholder(tf.float32,
            [None, len(actions.FUNCTIONS)], name='available_action_ids')
        self.ph_selected_spatial_action = tf.placeholder(tf.int32, [None, 2],
            name='selected_spatial_action')
        self.ph_selected_action_id = tf.placeholder(tf.int32, [None],
            name="selected_action_id")
        self.ph_value_target = tf.placeholder(tf.float32, [None], name='value_target')
        self.ph_player_relative_screen = tf.placeholder(tf.int32,
            [None, self.spatial_dim, self.spatial_dim], name="player_relative_screen")
        self.ph_player_relative_minimap = tf.placeholder(tf.int32,
            [None, self.spatial_dim, self.spatial_dim], name="player_relative_minimap")

    def _build_fullyconv_network(self):
        units_embedded = layers.embed_sequence(
            self.ph_screen_unit_type,
            vocab_size=SCREEN_FEATURES.unit_type.scale,
            embed_dim=self.unit_type_emb_dim,
            scope="unit_type_emb"
        )

        # Let's not one-hot zero which is background
        player_relative_screen_one_hot = layers.one_hot_encoding(
            self.ph_player_relative_screen,
            num_classes=SCREEN_FEATURES.player_relative.scale
        )[:, :, :, 1:]
        player_relative_minimap_one_hot = layers.one_hot_encoding(
            self.ph_player_relative_minimap,
            num_classes=MINIMAP_FEATURES.player_relative.scale
        )[:, :, :, 1:]

        channel_axis = 3
        screen_numeric_all = tf.concat(
            [self.ph_screen_numeric, units_embedded, player_relative_screen_one_hot],
            axis=channel_axis
        )
        minimap_numeric_all = tf.concat(
            [self.ph_minimap_numeric, player_relative_minimap_one_hot],
            axis=channel_axis
        )
        screen_output = _build_convs(screen_numeric_all, "screen_network")
        minimap_output = _build_convs(minimap_numeric_all, "minimap_network")

        map_output = tf.concat([screen_output, minimap_output], axis=channel_axis)

        spatial_action_logits = layers.conv2d(
            map_output,
            data_format="NHWC",
            num_outputs=1,
            kernel_size=1,
            stride=1,
            activation_fn=None,
            scope='spatial_action'
        )

        spatial_action_probs = tf.nn.softmax(layers.flatten(spatial_action_logits))

        map_output_flat = layers.flatten(map_output)

        fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1"
        )
        action_id_probs = layers.fully_connected(
            fc1,
            num_outputs=len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id"
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value'
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1
        action_id_probs *= self.ph_available_action_ids
        action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keep_dims=True)

        return spatial_action_probs, action_id_probs, value_estimate

    def build_model(self):
        self._define_input_placeholders()

        spatial_action_probs, action_id_probs, value_estimate = \
            self._build_fullyconv_network()

        selected_spatial_action_flat = ravel_index_pairs(
            self.ph_selected_spatial_action, self.spatial_dim
        )

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        spatial_action_log_probs = (
            logclip(spatial_action_probs)
            * tf.expand_dims(self.ph_is_spatial_action_available, axis=1)
        )

        # non-available actions get log(1e-10) value but that's ok because it's never used        
        action_id_log_probs = logclip(action_id_probs)

        selected_spatial_action_log_prob = select_from_each_row(
            spatial_action_log_probs, selected_spatial_action_flat
        )
        selected_action_id_log_prob = select_from_each_row(
            action_id_log_probs, self.ph_selected_action_id
        )
        selected_action_total_log_prob = (
            selected_spatial_action_log_prob
            + selected_action_id_log_prob
        )

        # maximum is to avoid 0 / 0 because this is used to calculate some means
        sum_spatial_action_available = tf.maximum(
            1e-10, tf.reduce_sum(self.ph_is_spatial_action_available)
        )
        neg_entropy_spatial = tf.reduce_sum(
            spatial_action_probs * spatial_action_log_probs
        ) / sum_spatial_action_available
        neg_entropy_action_id = tf.reduce_mean(tf.reduce_sum(
            action_id_probs * action_id_log_probs, axis=1
        ))

        advantage = tf.stop_gradient(self.ph_value_target - value_estimate)
        policy_loss = -tf.reduce_mean(selected_action_total_log_prob * advantage)
        value_loss = tf.losses.mean_squared_error(self.ph_value_target, value_estimate)

        loss = (
            policy_loss
            + value_loss * self.loss_value_weight
            + neg_entropy_spatial * self.entropy_weight_spatial
            + neg_entropy_action_id * self.entropy_weight_action_id
        )

        scalar_summary_collection_name = "scalar_summaries"
        s_collections = [scalar_summary_collection_name, tf.GraphKeys.SUMMARIES]
        tf.summary.scalar("loss/policy", policy_loss, collections=s_collections)
        tf.summary.scalar("loss/value", value_loss, s_collections)
        tf.summary.scalar("loss/neg_entropy_spatial", neg_entropy_spatial, s_collections)
        tf.summary.scalar("loss/neg_entropy_action_id", neg_entropy_action_id, s_collections)
        tf.summary.scalar("loss/total", loss, s_collections)
        tf.summary.scalar("value/advantage", tf.reduce_mean(advantage), s_collections)
        tf.summary.scalar("value/estimate", tf.reduce_mean(value_estimate), s_collections)
        tf.summary.scalar("value/target", tf.reduce_mean(self.ph_value_target), s_collections)
        tf.summary.scalar("action/is_spatial_action_available",
            tf.reduce_mean(self.ph_is_spatial_action_available), s_collections)
        tf.summary.scalar("action/is_spatial_action_available",
            tf.reduce_mean(self.ph_is_spatial_action_available), s_collections)
        tf.summary.scalar("action/selected_id_log_prob",
            tf.reduce_mean(selected_action_id_log_prob))
        tf.summary.scalar("action/selected_total_log_prob",
            tf.reduce_mean(selected_action_total_log_prob))
        tf.summary.scalar("action/selected_spatial_log_prob",
            tf.reduce_sum(selected_spatial_action_log_prob) / sum_spatial_action_available
        )

        self.sampled_action_id = weighted_random_sample(action_id_probs)
        self.sampled_spatial_action = weighted_random_sample(spatial_action_probs)
        self.value_estimate = value_estimate

        self.train_op = layers.optimize_loss(
            loss=loss,
            global_step=framework.get_global_step(),
            optimizer=self.optimiser,
            clip_gradients=self.max_gradient_norm,
            summaries=OPTIMIZER_SUMMARIES,
            learning_rate=None,
            name="train_op"
        )

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.all_summary_op = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)
        self.scalar_summary_op = tf.summary.merge(tf.get_collection(scalar_summary_collection_name))

    def obs_to_feeddict(self, obs):
        return {
            self.ph_minimap_numeric: obs["minimap_numeric"],
            self.ph_screen_numeric: obs["screen_numeric"],
            self.ph_available_action_ids: obs["available_actions"],
            self.ph_screen_unit_type: obs["screen_unit_type"],
            self.ph_player_relative_screen: obs["player_relative_screen"],
            self.ph_player_relative_minimap: obs["player_relative_minimap"]
        }

    def step(self, obs):
        feed_dict = self.obs_to_feeddict(obs)

        action_id, spatial_action = self.sess.run(
            [self.sampled_action_id, self.sampled_spatial_action],
            feed_dict=feed_dict
        )

        spatial_action_2d = np.array(
            np.unravel_index(spatial_action, (self.spatial_dim,) * 2)
        ).transpose()

        return action_id, spatial_action_2d

    def train(self,
            n_step_rewards,
            mb_obs_combined,
            mb_actions_combined
    ):
        feed_dict = {
            self.ph_value_target: n_step_rewards,
            self.ph_selected_spatial_action: mb_actions_combined["spatial_action"],
            self.ph_selected_action_id: mb_actions_combined["action_id"],
            self.ph_is_spatial_action_available: mb_actions_combined["is_spatial_action_available"]
        }
        feed_dict.update(self.obs_to_feeddict(mb_obs_combined))

        # treat each timestep as a separate observation, so batch_size will become batch_size * timesteps
        feed_dict = {k: combine_first_dimensions(v) for k, v in feed_dict.items()}
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
        feed_dict = self.obs_to_feeddict(obs)
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
