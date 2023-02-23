import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os
import reverb
import tempfile
import PIL.Image
import shutil
import datetime
import argparse

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_gym
#from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common
from tf_agents.policies import policy_saver

tempdir = tempfile.gettempdir()


#####     ARGUMENTS

parser = argparse.ArgumentParser(description='Description de votre programme')

parser.add_argument('--run_session_id', type=str, help='ID de la session en cours')
parser.add_argument('--env_name', type=str, default='Humanoid-v3', help='Nom de l\'environnement')
parser.add_argument('--num_iterations', type=int, default=51, help='Nombre d\'itérations')
parser.add_argument('--initial_collect_steps', type=int, default=10000, help='Nombre d\'étapes de collecte initiales')
parser.add_argument('--collect_steps_per_iteration', type=int, default=1, help='Nombre d\'étapes de collecte par itération')
parser.add_argument('--replay_buffer_capacity', type=int, default=10000, help='Capacité du buffer de replay')
parser.add_argument('--batch_size', type=int, default=256, help='Taille des lots')
parser.add_argument('--critic_learning_rate', type=float, default=3e-4, help='Taux d\'apprentissage du critique')
parser.add_argument('--actor_learning_rate', type=float, default=3e-4, help='Taux d\'apprentissage de l\'acteur')
parser.add_argument('--alpha_learning_rate', type=float, default=3e-4, help='Taux d\'apprentissage d\'alpha')
parser.add_argument('--target_update_tau', type=float, default=0.005, help='Tau de mise à jour de la cible')
parser.add_argument('--target_update_period', type=int, default=1, help='Période de mise à jour de la cible')
parser.add_argument('--gamma', type=float, default=0.99, help='Facteur de remise des récompenses')
parser.add_argument('--reward_scale_factor', type=float, default=1.0, help='Facteur d\'échelle des récompenses')
parser.add_argument('--actor_fc_layer_params', type=tuple, default=(256, 256), help='Paramètres des couches entièrement connectées de l\'acteur')
parser.add_argument('--critic_joint_fc_layer_params', type=tuple, default=(256, 256), help='Paramètres des couches entièrement connectées du critique')
parser.add_argument('--log_interval', type=int, default=5000, help='Intervalle de journalisation')
parser.add_argument('--save_interval', type=int, default=50000, help='Intervalle de sauvegarde')
parser.add_argument('--num_eval_episodes', type=int, default=20, help='Nombre d\'épisodes d\'évaluation')
parser.add_argument('--eval_interval', type=int, default=10000, help='Intervalle d\'évaluation')
parser.add_argument('--policy_save_interval', type=int, default=5000, help='Intervalle de sauvegarde de la politique')

args = parser.parse_args()

run_session_id = args.run_session_id
env_name = args.env_name
num_iterations = args.num_iterations
initial_collect_steps = args.initial_collect_steps
collect_steps_per_iteration = args.collect_steps_per_iteration
replay_buffer_capacity = args.replay_buffer_capacity
batch_size = args.batch_size
critic_learning_rate = args.critic_learning_rate
actor_learning_rate = args.actor_learning_rate
alpha_learning_rate = args.alpha_learning_rate
target_update_tau = args.target_update_tau
target_update_period = args.target_update_period
gamma = args.gamma
reward_scale_factor = args.reward_scale_factor
actor_fc_layer_params = args.actor_fc_layer_params
critic_joint_fc_layer_params = args.critic_joint_fc_layer_params
log_interval = args.log_interval
save_interval = args.save_interval
num_eval_episodes = args.num_eval_episodes
eval_interval = args.eval_interval
policy_save_interval = args.policy_save_interval

# Affichage des paramètres
print("Paramètres :")
print(f"run_session_id = {run_session_id}")
print(f"env_name = {env_name}")
print(f"num_iterations = {num_iterations}")
print(f"initial_collect_steps = {initial_collect_steps}")
print(f"collect_steps_per_iteration = {collect_steps_per_iteration}")
print(f"replay_buffer_capacity = {replay_buffer_capacity}")
print(f"batch_size = {batch_size}")
print(f"critic_learning_rate = {critic_learning_rate}")
print(f"actor_learning_rate = {actor_learning_rate}")
print(f"alpha_learning_rate = {alpha_learning_rate}")
print(f"target_update_tau = {target_update_tau}")
print(f"target_update_period = {target_update_period}")
print(f"gamma = {gamma}")
print(f"reward_scale_factor = {reward_scale_factor}")
print(f"actor_fc_layer_params = {actor_fc_layer_params}")
print(f"critic_joint_fc_layer_params = {critic_joint_fc_layer_params}")
print(f"log_interval = {log_interval}")
print(f"save_interval = {save_interval}")
print(f"num_eval_episodes = {num_eval_episodes}")
print(f"eval_interval = {eval_interval}")
print(f"policy_save_interval = {policy_save_interval}")

#####    INIT

model_name = "humanoid-arg-py"
if not os.path.isdir("/app/" + model_name):
    os.makedirs("/app/" + model_name)
session_dir = "/app/" + model_name + "/" + str(run_session_id)

os.makedirs(session_dir)


env = suite_gym.load(env_name)
env.reset()


collect_env = suite_gym.load(env_name)
eval_env = suite_gym.load(env_name)

#TRAIN WITH GPU
use_gpu = False #@param {type:"boolean"}
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)


#Agents

observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))

with strategy.scope():
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

with strategy.scope():
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))


with strategy.scope():
    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.keras.optimizers.Adam(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

    tf_agent.initialize()
    
    
    
  
    #Tampon
rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)
table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])
reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)

#Dataset
dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset



#Stratégies
tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_eval_policy, use_tf_function=True)
tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
    tf_collect_policy, use_tf_function=True)

random_policy = random_py_policy.RandomPyPolicy(
    collect_env.time_step_spec(), collect_env.action_spec())


#Acteurs
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    reverb_replay.py_client,
    table_name,
    sequence_length=2,
    stride_length=1)

initial_collect_actor = actor.Actor(
    collect_env,
    random_policy,
    train_step,
    steps_per_run=initial_collect_steps,
    observers=[rb_observer])
initial_collect_actor.run()



#Actor Collect and Form 
env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
    collect_env,
    collect_policy,
    train_step,
    steps_per_run=1,
    metrics=actor.collect_metrics(10),
    summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
    observers=[rb_observer, env_step_metric])



eval_actor = actor.Actor(
    eval_env,
    eval_policy,
    train_step,
    episodes_per_run=num_eval_episodes,
    metrics=actor.eval_metrics(num_eval_episodes),
    summary_dir=os.path.join(tempdir, 'eval'),
    )


#Apprenants
saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers,
  strategy=strategy)



#Metrics and eval
def get_eval_metrics():
    eval_actor.run()
    results = {}
    for metric in eval_actor.metrics:
        results[metric.name] = metric.result()
    return results

def log_eval_metrics(step, metrics):
    eval_results = (', ').join(
        '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print('step = {0}: {1}'.format(step, eval_results))
    
metrics = get_eval_metrics()
log_eval_metrics(0, metrics)



global_step = tf.compat.v1.train.get_or_create_global_step()



checkpoint_dir = os.path.join(session_dir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=10,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=reverb_replay,
    global_step=global_step
)

policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)


def create_zip_file(dirname, base_filename):
    return shutil.make_archive(base_filename, 'zip', dirname)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
metrics_tmp = get_eval_metrics()
metrics_result = {}
for m in metrics_tmp:
    metrics_result[m] = [metrics_tmp[m]]
  
# using now() to get current time

#Video
def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


for i in range(num_iterations):
    # Training.
    collect_actor.run()
    loss_info = agent_learner.run(iterations=1)

    # Evaluating.
    step = agent_learner.train_step_numpy

    if step % eval_interval == 0:
        metrics = get_eval_metrics()
        log_eval_metrics(step, metrics)
        f = open(session_dir + "/log.txt", "a")
        for m in metrics:
            metrics_result[m].append(metrics[m])
            f.write(str(step) + ";" + m + ";" + str(metrics[m]) + "\n")
        f.close()
      
        
    if step % log_interval == 0:
        f = open(session_dir + "/log.txt", "a")
        f.write(str(step) + ";loss;" + str(loss_info.loss.numpy()) + "\n")
        f.close()
        print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))
        
    if step % save_interval ==0 :
        train_checkpointer.save(global_step)
        checkpoint_zip_filename = create_zip_file(checkpoint_dir, os.path.join('.', 'exported_cp_humanoid'))
        
        steps = range(0, i + 2, eval_interval)
        for m in metrics_result:
            plt.clf()
            plt.plot(steps, metrics_result[m])
            plt.ylabel(m)
            plt.xlabel('Step')
            plt.ylim()
            plt.savefig(session_dir + "/result_" + m + "-" + str(step) + ".png")
        
        num_episodes = num_iterations
        video_filename = session_dir + '/humanoid-' + str(i) + '.mp4'
        with imageio.get_writer(video_filename, fps=60) as video:
            time_step = eval_env.reset()
            video.append_data(eval_env.render())
            while not time_step.is_last():
                action_step = eval_actor.policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_env.render())
        embed_mp4(video_filename)

        
rb_observer.close()
reverb_server.stop()