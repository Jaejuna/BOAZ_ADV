import ray
from ray.rllib.algorithms.ppo import PPOTrainer, PPOConfig
from ray import tune

import gym
from gym import spaces

import airsim

import numpy as np

from tools.test import getAccuracy
from tools.models import MovePredictModel
from tools.vision import *
from tools.train_utils import * 
from tools.live_visualization import live_visualization

from config.default import args

class CustomAirSim(gym.Env):
    def __init__(self, config=None):
        # AirSim 클라이언트와 기타 필요한 변수를 초기화
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True) # 드론의 시동을 걸어줌, 가상 드론의 모터를 활성화하고 비행 준비 상태로 전환
        self.client.takeoffAsync().join() # 이륙, join()은 이륙 작업이 완료될 때까지 기다리는 역할
        self.client.hoverAsync().join() # 현재위치에서 정지비행상태로 전환, join은 완료할 때까지 기다리는 역할

        # 행동 및 관찰 공간을 정의합니다.
        self.action_space = spaces.Discrete(8)  # [앞,뒤,좌,우,위,아래,왼회전,오른회전]? '행동 세부 정의 필요'
        self.observation_space = spaces.Box(low=0, high=255, shape=(...), dtype=np.uint8)  # '원하는 이미지 정의 필요'
        
        # ppo 알고리즘 초기화
        self.init_ppo_config()
        
    def init_ppo_config(self):
        self.ppo_config = PPOConfig()
        # kl_coeff, ->
        # vf_loss_coeff used to be 0.01??
        # "entropy_coeff": 0.00005,
        # "clip_param": 0.1,
        self.ppo_config.gamma = 0.998  # default 0.99
        self.ppo_config.lambda_ = 0.99  # default 1.0???
        self.ppo_config.kl_target = 0.01  # used to use 0.02
        self.ppo_config.rollout_fragment_length = 512
        self.ppo_config.train_batch_size = 6400
        self.ppo_config.sgd_minibatch_size = 256
        self.ppo_config.num_sgd_iter = 2 # default 30???
        self.ppo_config.lr = 3.5e-5  # 5e-5
        self.ppo_config.model = {
            # Share layers for value function. If you set this to True, it's
            # important to tune vf_loss_coeff.
            "vf_share_layers": False,

            "use_lstm": True,
            "max_seq_len": 32,
            "lstm_cell_size": 128,
            "lstm_use_prev_action": True,

            # 'use_attention': True,
            # "max_seq_len": 128,
            # "attention_num_transformer_units": 1,
            # "attention_dim": 1024,
            # "attention_memory_inference": 128,
            # "attention_memory_training": 128,
            # "attention_num_heads": 8,
            # "attention_head_dim": 64,
            # "attention_position_wise_mlp_dim": 512,
            # "attention_use_n_prev_actions": 0,
            # "attention_use_n_prev_rewards": 0,
            # "attention_init_gru_gate_bias": 2.0,

            "conv_filters": [
                # [4, [3, 4], [1, 1]],
                # [16, [6, 8], [3, 3]],
                # [32, [6, 8], [3, 4]],
                # [64, [6, 6], 3],
                # [256, [9, 9], 1],

                # 480 x 640
                # [4, [7, 7], [3, 3]],
                # [16, [5, 5], [3, 3]],
                # [32, [5, 5], [2, 2]],
                # [64, [5, 5], [2, 2]],
                # [256, [5, 5], [3, 5]],

                # 240 X 320
                [16, [5, 5], 3],
                [32, [5, 5], 3],
                [64, [5, 5], 3],
                [128, [3, 3], 2],
                [256, [3, 3], 2],
                [512, [3, 3], 2],
            ],
            "conv_activation": "relu",
            "post_fcnet_hiddens": [512],
            "post_fcnet_activation": "relu"
        }
        self.ppo_config.batch_mode = "complete_episodes"
        self.ppo_config.simple_optimizer = True
        self.ppo_config.num_gpus = 1

        self.ppo_config.rollouts(num_rollout_workers=0)

        self.ppo_config.env = None
        self.ppo_config.observation_space = spaces.Box(low=0, high=1, shape=('y','x', 1), dtype=np.float32) # 원하는 상태(이미지) 데이터 형태 정의 필요
        self.ppo_config.action_space = spaces.MultiDiscrete(    # agent action 가짓수 정의 필요 
            [
                2,  # W
                2,  # A
                2,  # S
                2,  # D
                2,  # Space
                2,  # H
                2,  # J
                2,  # K
                2  # L
            ]
        )
        self.ppo_config.env_config = {
            "sleep": True,
        }
        self.ppo_config.framework_str = 'tf'
        self.ppo_config.log_sys_usage = False
        self.ppo_config.compress_observations = True
        self.ppo_config.shuffle_sequences = False
        
        return self.ppo_config    
    
    # 필요한 경우 PPO 설정을 업데이트하는 메서드
    #def update_ppo_config(self, new_config):
    #    for key, value in new_config.items():
    #        setattr(self.ppo_config, key, value)

    def reset(self):
        # 환경을 초기 상태로 리셋하고 초기 관찰값을 반환합니다.
        self.client.reset()
        return self._get_observation()

    def step(self, action):
        # 행동을 실행하고, 다음 상태, 보상, 종료 여부, 추가 정보를 반환합니다.
        reward = 0
        done = False
        info = {}

        # ... 행동 실행 ...

        observation = self._get_observation()
        return observation, reward, done, info

    def _get_observation(self):
        # 현재 상태 (예: RGB 이미지)를 반환하는 도우미 함수
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        return img_rgb

    def close(self):
        # 환경을 종료하기 전에 호출됩니다.
        self.client.enableApiControl(False)
        
if __name__ == "__main__":
    
    trainer = PPOTrainer(env=CustomAirSim, config={
        "env_config": {},
        # 다른 설정 ...
    })

    ray.init()
    # ray.init(num_cpus=4, num_gpus=1, log_to_driver=False) #init example
    trainer = PPOTrainer
    env = CustomAirSim()
    
    tune.run(trainer,
            resume='AUTO',
            config=env.ppo_config.to_dict(), keep_checkpoints_num=None, checkpoint_score_attr="episode_reward_mean",
            max_failures=1,
            checkpoint_freq=5, checkpoint_at_end=True)