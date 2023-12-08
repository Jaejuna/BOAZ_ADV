from easydict import EasyDict as edict
import torch

drone = edict()
drone.yaw_rate = 90
drone.default_velocity = 5
drone.moving_unit = 2
drone.set_random_pose = False
drone.x_range = 100
drone.y_range = 100
drone.z_range = 5
drone.MIN_DEPTH_METERS = 0
drone.MAX_DEPTH_METERS = 100

# main
args = edict(drone=drone)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.map_path = "./maps/map_created_by_lidar_small_filtered.ply"

args.num_classes = 4

args.gamma = 0.9
args.learning_rate = 1e-3
args.epsilon = 0.5
args.min_epsilon = 0.01
args.epsilon_decay_rate = 0.95

args.checkpoint = None ## Load chekpoint

args.batch_size = 4 # batch size
args.mem_size = 50
args.epochs = 5000  
args.max_time = 180  
args.sync_freq = 500

args.eval_freq = 50

args.voxel_threshold = 0.002
args.reward_state_threshold = 0.5
args.voxel_size = 0.5

args.backbone_name = "resnet152"

args.eval_steps = 10

args.live_visualization = False