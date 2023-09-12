from easydict import EasyDict as edict
import torch

drone = edict()
drone.yaw_rate = 90
drone.default_velocity = 5
drone.moving_unit = 0.5
drone.set_random_pose = True
drone.x_range = 1000
drone.y_range = 1000
drone.z_range = 5
drone.MIN_DEPTH_METERS = 0
drone.MAX_DEPTH_METERS = 100

# main
args = edict(drone=drone)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.num_classes = 4

args.gamma = 0.9
args.learning_rate = 1e-3
args.epsilon = 0.3 

args.checkpoint = None ## Load chekpoint

args.batch_size = 4 # batch size
args.mem_size = 50
args.epochs = 5000  
args.max_time = 60  
args.sync_freq = 500

args.eval_freq = 50

args.voxel_threshold = 0.002
args.voxel_size = 0.5

args.model_name = ""

args.eval_steps = 100