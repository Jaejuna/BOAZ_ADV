import utils.setup_path as setup_path
import airsim

from math import pi
import numpy as np
import torch.nn.functional as F
import random
import time

from tools.vision import *
from tools.train_utils import *

def clacDuration(yaw_rate, radian):
    duration = (pi / radian) / (yaw_rate * pi / 180)
    return duration

def calcValues(qval, client, args):
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

    pose = client.simGetVehiclePose()
    action_vector[0] += pose.position.x_val
    action_vector[1] += pose.position.y_val
    action_vector[2] -= pose.position.z_val

    if radian < -1:  radian %= -1
    elif radian > 1: radian %= 1
    
    return *action_vector, radian, action

def calcStatus(reward):
    status = "running"
    if abs(reward) != 2:
        if reward > 0:
            status = "work well"
        else:
            status = "work bad"
    return status

def calcReward(map_voxel, map_info, prev_pcd, curr_pcd, client, running_time, args):

    # 상태에서 필요한 정보 추출
    collision_info = client.simGetCollisionInfo()

    # 보상 초기값
    reward = 0.0

    # 이전 포인트 클라우드와 현재 포인트 클라우드의 차이 계산
    prev_voxel, _ = pcd_to_voxel_tensor(prev_pcd, infos=map_info)
    curr_voxel, _ = pcd_to_voxel_tensor(prev_pcd + curr_pcd, infos=map_info)
    
    # 두 포인트 클라우드의 크기를 동일하게 맞춤
    mse_mc = F.mse_loss(map_voxel, curr_voxel, reduction='mean').item()
    mse_pc = F.mse_loss(prev_voxel, curr_voxel, reduction='mean').item()
    print("mse_mc:", mse_mc)
    print("mse_pc:", mse_pc)
    
    # 드론이 충돌한 경우
    if collision_info.has_collided:
        reward -= 20.0  # 충돌 시 큰 음수 값으로 보상
        print("드론이 충돌한 경우")
    elif args.max_time >= running_time:
        reward -= 10.0
        print("시간 초과")

    # 전에 만든 맵과 현재 만든 맵의 차이에 따른 보상
    if mse_pc >= args.voxel_threshold:
        reward -= 1.0
        print("전에 만든 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상")
    elif mse_pc < args.voxel_threshold:
        reward -= 10.0
        print("전에 만든 맵과 현재 만든 맵의 차이가 작으면 큰 음의 보상")

    # 전체 맵과 현재 만든 맵의 차이에 따른 보상
    if mse_mc >= args.voxel_threshold:
        reward -= 1.0
        print("전체 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상")
    elif mse_mc < args.voxel_threshold:
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

def putDataIntoQueue(data_queue, pcd):
    data = {
        "points" : list(pcd.points),
        "colors" : list(pcd.colors)
    }
    data_queue.put(data)

def connectToClient():
    client = airsim.MultirotorClient() # airsim 클라이언트 객체 생성
    client.confirmConnection() # 클라이언트와 시뮬레이션서버간의 연결 확인, 연결이 안되어있으면 에러 발생
    client.enableApiControl(True) # 드론의 API 제어를 활성화, 활성화 해야 드론에 명령을 내릴 수 있음, 비활성화시 드론은 움직이지 않음.
    return client

def droneReadyState(client):
    client.armDisarm(True) # 드론의 시동을 걸어줌, 가상 드론의 모터를 활성화하고 비행 준비 상태로 전환
    client.takeoffAsync().join() # 이륙, join()은 이륙 작업이 완료될때까지 기다리는 역할을 함
    client.hoverAsync().join() # 드론이 현재위치에서 정지비행상태로 전환하는 메서드, join은 완료할 때까지 기다리는 역할
    return client

def resetState(client, data_queue):
    print("\nreset client...")
    client.reset()
    data_queue.put("reset")
    time.sleep(2)
    client.confirmConnection() # 클라이언트와 시뮬레이션서버간의 연결 확인, 연결이 안되어있으면 에러 발생
    client.enableApiControl(True) # 드론의 API 제어를 활성화, 활성화 해야 드론에 명령을 내릴 수 있음, 비활성화시 드론은 움직이지 않음.
    client.armDisarm(True) # 드론의 시동을 걸어줌, 가상 드론의 모터를 활성화하고 비행 준비 상태로 전환
    client.takeoffAsync().join() # 이륙, join()은 이륙 작업이 완료될때까지 기다리는 역할을 함
    client.hoverAsync().join() # 드론이 현재위치에서 정지비행상태로 전환하는 메서드, join은 완료할 때까지 기다리는 역할
    return client