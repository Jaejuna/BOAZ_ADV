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
    action_vector = qval[0, :3]
    one_hot = np.zeros_like(action_vector)
    action = np.argmax(action_vector)
    radian = qval[0, 3]

    if random.random() < args.epsilon:
        action = np.random.randint(0, args.num_classes - 1)
        radian = random.uniform(-1, 1)
        
    one_hot[action] = 1
    action_vector = one_hot * args["drone"].moving_unit
    action_vector = list(map(float, action_vector))

    if radian < -1:  radian %= -1
    elif radian > 1: radian %= 1
    

    return *action_vector, radian, action

def calcStatus(reward, elapsedTime, args):
    status = "running"
    if args.max_time > elapsedTime:
        status = "time out"
        if reward != -1:
            if reward > 0:
                status = "work well"
            else:
                status = "work bad"
    return status

def calcReward(map_pcd, prev_pcd, curr_pcd, client, args):

    # 상태에서 필요한 정보 추출
    collision_info = client.simGetCollisionInfo()

    # 보상 초기값
    reward = 0.0

    # 이전 포인트 클라우드와 현재 포인트 클라우드의 차이 계산
    map_np = np.asarray(map_pcd.points)
    prev_np = np.asarray(prev_pcd.points)
    curr_np = np.asarray(curr_pcd.points)
    
    # 두 포인트 클라우드의 크기를 동일하게 맞춤
    min_points_mc = min(map_np.shape[0], curr_np.shape[0])
    mse_mc = np.mean((map_np[:min_points_mc] - curr_np[:min_points_mc]) ** 2)
    min_points_pc = min(prev_np.shape[0], curr_np.shape[0])
    mse_pc = np.mean((prev_np[:min_points_pc] - curr_np[:min_points_pc]) ** 2)
    print("mse_mc:", mse_mc)
    print("mse_pc:", mse_pc)
    
    # 만약 드론이 충돌한 경우
    if collision_info.has_collided:
        reward -= -10.0  # 큰 음수 값으로 보상
        print("드론이 충돌한 경우")
    # 전에 만든 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상
    # 계속해서 진행하도록
    if mse_pc >= args.voxel_threshold:
        reward -= 1.0
        print("전에 만든 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상")
    # 전에 만든 맵과 현재 만든 맵의 차이가 작으면 큰 음의 보상
    # 드론이 조금 움직인 것이므로 
    elif mse_pc < args.voxel_threshold:
        reward -= 10.0
        print("전에 만든 맵과 현재 만든 맵의 차이가 작으면 큰 음의 보상")
    # 전체 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상
    # 계속해서 진행하도록
    elif mse_mc >= args.voxel_threshold:
        reward -= 1.0
        print("전체 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상")
    # 전체 맵과 현재 만든 맵의 차이가 작으면 큰 양의 보상
    # 전체 맵을 잘 만든 것이므로
    if mse_mc < args.voxel_threshold:
        reward = 10.0
        print("전체 맵과 현재 만든 맵의 차이가 작으면 큰 양의 보상")


    print("최종 reward:", reward)

    return reward

def setRandomPose(client, args):
    collision = True
    while collision:
        # 임의의 위치와 방향 생성
        x_range = args["drone"].x_range
        y_range = args["drone"].y_range
        z_range = args["drone"].z_range
        
        x, y, z = random.uniform(-x_range, x_range), random.uniform(-y_range, y_range), random.uniform(0, z_range)
        pitch, roll, yaw = 0, 0, random.uniform(0, 360)  # 예시 값

        pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch, roll, yaw))
        client.simSetVehiclePose(pose, True)
        
        # 충돌 정보 확인
        collision_info = client.simGetCollisionInfo()
        collision = collision_info.has_collided