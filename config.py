from easydict import EasyDict as edict
import torch

drone = edict()
drone.yaw_rate = 90
drone.default_velocity = 5

# main
args = edict(drone=drone)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.exp_time = None
args.exp_name = None

args.n_classes = 4

args.gamma = 0.9
args.learning_rate = 1e-3
args.epsilon = 0.3 

args.checkpoint = None ## Load chekpoint

args.batch_size = 200 # batch size
args.mem_size = 1000
args.epochs = 5000  
args.max_time = 60  
args.sync_freq = 500

args.save_dir = ""