import argparse

parser = argparse.ArgumentParser(description='Description de votre programme')

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

# Récupération des paramètres passés en ligne de commande
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
