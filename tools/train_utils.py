import utils.setup_path as setup_path
import airsim

from math import pi
import numpy as np
import random
import torch

from tools.vision import *
from tools.train_utils import *

def clacDuration(yaw_rate, radian):
    duration = (pi / radian) / (yaw_rate * pi / 180)
    return duration

def calcValues(qval, args):
    action_vector = qval[:3]
    one_hot = np.zeros_like(action_vector)
    action = np.argmax(action_vector)
    radian = qval[3]

    if random.random() < args.epsilon:
        action = np.random.randint(0, args.n_classes - 1)
        radian = random.uniform(-1, 1)
        
    one_hot[action] = 1
    action_vector = one_hot * args["drone"].moving_unit

    if radian < -1:  radian %= -1
    elif radian > 1: radian %= 1

    return *action_vector, radian, action

def calcReward(map_pcd, prev_pcd, curr_pcd, client, args):

    # 상태에서 필요한 정보 추출
    collision_info = client.collision_info

    # 보상 초기값
    reward = 0.0
    
    # 만약 드론이 충돌한 경우
    if collision_info.has_collided:
        reward = -10.0  # 큰 음수 값으로 보상

    # 이전 포인트 클라우드와 현재 포인트 클라우드의 차이 계산
    map_np = np.asarray(map_pcd.points)
    prev_np = np.asarray(prev_pcd.points)
    curr_np = np.asarray(curr_pcd.points)
    
    # 두 포인트 클라우드의 크기를 동일하게 맞춤
    min_points_mc = min(map_np.shape[0], curr_np.shape[0])
    mse_mc = np.mean((map_np[:min_points_mc] - curr_np[:min_points_mc]) ** 2)
    min_points_pc = min(prev_np.shape[0], curr_np.shape[0])
    mse_pc = np.mean((prev_np[:min_points_pc] - curr_np[:min_points_pc]) ** 2)
    
    # 전체 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상
    # 계속해서 진행하도록
    if mse_mc > args.voxel_threshold:
        reward -= 1.0
    # 전체 맵과 현재 만든 맵의 차이가 작으면 큰 양의 보상
    # 전체 맵을 잘 만든 것이므로
    elif mse_mc < args.voxel_threshold:
        reward = 10.0
    # 전에 만든 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상
    # 계속해서 진행하도록
    elif mse_pc > args.voxel_threshold:
        reward -= 1.0
    # 전에 만든 맵과 현재 만든 맵의 차이가 작으면 큰 음의 보상
    # 드론이 조금 움직인 것이므로 
    elif mse_pc < args.voxel_threshold:
        reward -= 10.0

    return reward