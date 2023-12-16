from easydict import EasyDict as edict
import torch

drone = edict()
drone.yaw_rate = 90
drone.default_velocity = 2
drone.moving_unit = 2
drone.set_random_pose = False
drone.x_range = 100
drone.y_range = 100
drone.z_range = 5
drone.MIN_DEPTH_METERS = 0
drone.MAX_DEPTH_METERS = 100

TDRS = edict()
TDRS.decay_factor = 0.95
TDRS.min_decay_factor = 0
TDRS.every_second = 10

manual = edict()
manual.txt_path = "data/name_tags.txt"
manual.batch_size = 16

# main
args = edict(drone=drone, TDRS=TDRS, manual=manual)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.map_path = "./maps/map_created_by_lidar_small_filtered_without_floor2.ply"

args.checkpoint = None ## Load chekpoint

args.num_classes = 3
args.backbone_name = "resnet152"
args.sync_freq = 500

args.gamma = 0.9
args.learning_rate = 1e-4
args.epsilon = 0.5
args.min_epsilon = 0.01
args.epsilon_decay_rate = 0.95

args.batch_size = 4 # batch size
args.mem_size = 100
args.epochs = 5000  
args.max_time = 300

args.voxel_threshold = 0.002
args.reward_state_threshold = 0.5
args.voxel_size = 0.5

args.eval_freq = 50
args.eval_steps = 10

args.live_visualization = False

args.train_mode = "manual" # manual or auto