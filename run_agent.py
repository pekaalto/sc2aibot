import logging
import sys
import os
import shutil
import sys
from datetime import datetime
from functools import partial
import tensorflow as tf
from absl import flags
from actorcritic.agent import ActorCriticAgent, ACMode
from actorcritic.runner import Runner, PPORunParams
from common.multienv import SubprocVecEnv, make_sc2env, SingleEnv

FLAGS = flags.FLAGS
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 1, "Number of environments to run in parallel")
flags.DEFINE_integer("n_steps_per_batch", None,
    "Number of steps per batch, if None use 8 for a2c and 128 for ppo")
flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
flags.DEFINE_string("model_name", "temp_testing", "Name for checkpoints and tensorboard summaries")
flags.DEFINE_integer("K_batches", -1,
    "Number of training batches to run in thousands, use -1 to run forever")
flags.DEFINE_string("map_name", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
flags.DEFINE_boolean("training", True,
    "if should train the model, if false then save only episode score summaries"
)
flags.DEFINE_enum("if_output_exists", "fail", ["fail", "overwrite", "continue"],
    "What to do if summary and model output exists, only for training, is ignored if notraining")
flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")
flags.DEFINE_float("loss_value_weight", 1.0, "good value might depend on the environment")
flags.DEFINE_float("entropy_weight_spatial", 1e-6,
    "entropy of spatial action distribution loss weight")
flags.DEFINE_float("entropy_weight_action", 1e-6, "entropy of action-id distribution loss weight")
flags.DEFINE_float("ppo_lambda", 0.95, "lambda parameter for ppo")
flags.DEFINE_integer("ppo_batch_size", None, "batch size for ppo, if None use n_steps_per_batch")
flags.DEFINE_integer("ppo_epochs", 3, "epochs per update")
flags.DEFINE_enum("agent_mode", ACMode.A2C, [ACMode.A2C, ACMode.PPO], "if should use A2C or PPO")

FLAGS(sys.argv)

# TODO below it gets little messy with the folders, maybe do something more clever

full_chekcpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name)

if FLAGS.training:
    full_summary_path = os.path.join(FLAGS.summary_path, FLAGS.model_name)
else:
    full_summary_path = os.path.join(FLAGS.summary_path, "no_training", FLAGS.model_name)


def check_and_handle_existing_folder(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)


if FLAGS.training:
    check_and_handle_existing_folder(full_chekcpoint_path)
    check_and_handle_existing_folder(full_summary_path)

env_args = dict(
    map_name=FLAGS.map_name,
    step_mul=FLAGS.step_mul,
    game_steps_per_episode=0,
    screen_size_px=(FLAGS.resolution,) * 2,
    minimap_size_px=(FLAGS.resolution,) * 2,
    visualize=FLAGS.visualize
)

envs = SubprocVecEnv((partial(make_sc2env, **env_args),) * FLAGS.n_envs)
# envs = SingleEnv(make_sc2env(**env_args))

tf.reset_default_graph()
sess = tf.Session()

agent = ActorCriticAgent(
    mode=FLAGS.agent_mode,
    sess=sess,
    spatial_dim=FLAGS.resolution,
    unit_type_emb_dim=5,
    loss_value_weight=FLAGS.loss_value_weight,
    entropy_weight_action_id=FLAGS.entropy_weight_action,
    entropy_weight_spatial=FLAGS.entropy_weight_spatial,
    scalar_summary_freq=FLAGS.scalar_summary_freq,
    all_summary_freq=FLAGS.all_summary_freq,
    summary_path=full_summary_path,
    max_gradient_norm=FLAGS.max_gradient_norm
)

agent.build_model()
if os.path.exists(full_chekcpoint_path):
    agent.load(full_chekcpoint_path)
else:
    agent.init()

if FLAGS.n_steps_per_batch is None:
    n_steps_per_batch = 128 if FLAGS.agent_mode == ACMode.PPO else 8
else:
    n_steps_per_batch = FLAGS.n_steps_per_batch

if FLAGS.agent_mode == ACMode.PPO:
    ppo_par = PPORunParams(
        FLAGS.ppo_lambda,
        batch_size=FLAGS.ppo_batch_size or n_steps_per_batch,
        n_epochs=FLAGS.ppo_epochs
    )
else:
    ppo_par = None

runner = Runner(
    envs=envs,
    agent=agent,
    discount=FLAGS.discount,
    n_steps=n_steps_per_batch,
    do_training=FLAGS.training,
    ppo_par=ppo_par
)

runner.reset()

if FLAGS.K_batches >= 0:
    n_batches = FLAGS.K_batches * 1000
else:
    n_batches = -1


def _print(i):
    print(datetime.now())
    print("# batch %d" % i)
    sys.stdout.flush()


def _save_if_training():
    if FLAGS.training:
        agent.save(full_chekcpoint_path)
        agent.flush_summaries()
        sys.stdout.flush()


i = 0

try:
    while True:
        if i % 500 == 0:
            _print(i)
        if i % 4000 == 0:
            _save_if_training()
        runner.run_batch()
        i += 1
        if 0 <= n_batches <= i:
            break
except KeyboardInterrupt:
    pass

print("Okay. Work is done")
_print(i)
_save_if_training()

envs.close()
