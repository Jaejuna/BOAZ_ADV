import utils.setup_path as setup_path
import airsim

from math import pi
import numpy as np
import random
import torch

from vision import *
from train_utils import *

def test_model(model, client, map_pcd, args):
    pcd_global = o3d.geometry.PointCloud()
    pcd_global = merge_point_clouds(pcd_global, getPointCloud(client), client)

    rgb1 = getRGBImage(client)
    rgb1 = torch.from_numpy(rgb1).to(args.device)

    status = "running"
    start_time = time.time()
    while(status == "running"): 
        qval = model(rgb1)
        qval = qval.data.numpy()

        x, y, z, radian, _ = calcValues(qval, args.epsilon)
        
        client.moveToPositionAsync(x, y, z, args["drone"].defualt_velocity, 
                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                   yaw_mode=airsim.YawMode(False, clacDuration(args["drone"].yaw_rate, radian))).join()

        curr_pcd = merge_point_clouds(pcd_global, getPointCloud(client), client)
        reward = calcReward(map_pcd, pcd_global, curr_pcd, client)
        pcd_global = copy.deepcopy(curr_pcd)

        rgb2 = getRGBImage(client)
        rgb2 = torch.from_numpy(rgb2).to(args.device)
        rgb1 = rgb2

        if reward != -1:
            if reward > 0:
                status = "success"
            else:
                status = "fail"
            
        if args.max_time > (time.time() - start_time):
            break

    SlamWell = True if status == "success" else False
    return SlamWell

def getAccuracy(model, client, map_pcd, args, max_games=10000):
    SlamWells = 0
    for _ in range(max_games):
        SlamWell = test_model(model, client, map_pcd, args)
        if SlamWell:
            SlamWells += 1
    SlamWell_perc = float(SlamWells) / float(max_games)
    print("Simulated times: {0}, # of SlamWell percentage: {1}".format(max_games, 100.0 * SlamWells))
    return 100.0 * SlamWell_perc 

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