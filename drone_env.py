import gym
from gym import spaces
import airsim

import time
import numpy as np

from tools.vision import *
from tools.train_utils import * 

class AirSimEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape):
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, logger, args):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.args = args
        self.logger = logger

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(3)
        self._setup_flight()
        self.initial_z = self.drone.getMultirotorState().kinematics_estimated.position.z_val

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

        self.global_pcd = o3d.geometry.PointCloud()
        self.map_pcd_len = len(getMapPointCloud(args).points)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone = resetState(self.drone)
        self.running_time = 0

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        img2d = np.expand_dims(img2d, axis=-1)
        return img2d

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        
        self.curr_pcd = get_transformed_lidar_pc(self.drone)
        global_pcd = self.global_pcd + self.curr_pcd    
        self.global_pcd = global_pcd.voxel_down_sample(voxel_size=self.args.voxel_size)

        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        self.state["collision"] = self.drone.simGetCollisionInfo().has_collided

        return image

    def _do_action(self, action):            
        if action == 0:
            current_z = self.drone.getMultirotorState().kinematics_estimated.position.z_val
            v = args["drone"].default_velocity
            x, y, _ = calcForwardDirection(self.drone)
            z = (self.initial_z - current_z) * 0.5 
            move_with_timeout(self.drone, x * v, y * v, z, 1) 
        else:
            rotate_with_timeout(self.drone, calcDegree(self.drone, action), 1) 

    def _compute_reward(self):
        reward = calcReward(self.global_pcd, self.curr_pcd, self.args) 
        done = len(self.global_pcd.points) >= self.map_pcd_len * 0.7
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        print_and_logging(self.logger, f"reward, done, self.state : {reward} {done} {self.state}")
        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()