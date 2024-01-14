from easydict import EasyDict as edict
import torch
from torchvision import transforms

drone = edict()
drone.default_velocity = 2

# main
args = edict(drone=drone)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.map_path = "maps/map_created_by_lidar_small_filtered_without_floor2.ply"

args.learning_rate = 0.00025
args.batch_size = 32 # batch size
args.buffer_size = 500000
args.voxel_size = 1
args.verbose = 1
args.train_freq = 4
args.target_update_interval = 10000
args.learning_starts = 10000
args.max_grad_norm = 10
args.exploration_fraction = 0.1
args.exploration_final_eps = 0.01

args.callback_on_new_best = None
args.n_eval_episodes = 5
args.eval_freq = 10000

args.total_timesteps = 5e5
