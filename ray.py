# import setup_path
import gym
import airgym

import os
from datetime import datetime
import time

from drone_env import AirSimDroneEnv
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from tools.train_utils import * 
from config.default import args

date = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
job_dir = os.path.join("./run", date)
os.makedirs(job_dir, exist_ok=True)

logger = make_logger(job_dir)

# Create a DummyVecEnv for main airsim gym env
env = AirSimDroneEnv(
    ip_address="127.0.0.1",
    step_length=0.25,
    image_shape=(224, 224, 1),
    logger=logger,
    args=args
)

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=500000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log=job_dir,
)

# model = A2C(
#     "CnnPolicy",
#     env,
#     learning_rate=7e-4,
#     n_steps=5,
#     gamma=0.99,
#     gae_lambda=1.0,
#     ent_coef=0.0,
#     n_envs=4,
#     vf_coef=0.5,
#     max_grad_norm=0.5,
#     use_rms_prop=True,
#     tensorboard_log=job_dir,
#     verbose=1,
#     device="cuda"
# )

# model = PPO(
#     "CnnPolicy",
#     env,
#     learning_rate=3e-4,
#     n_steps=2048,
#     batch_size=32,
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.2,
#     ent_coef=0.0,
#     vf_coef=0.5,
#     max_grad_norm=0.5,
#     verbose=1,
#     tensorboard_log=job_dir,
#     device="cuda"
# )

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=job_dir,
    log_path=job_dir,
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5,
    tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("dqn_airsim_drone_policy")
