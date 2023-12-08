import utils.setup_path as setup_path
import airsim

from math import pi
import numpy as np
import torch.nn.functional as F
import random
import time

from tools.vision import *
from tools.train_utils import *

import threading

import logging

class StoppableThread(threading.Thread):
    def __init__(self, target, args=(), kwargs=None):
        super().__init__(target=target, args=args, kwargs=kwargs if kwargs else {})
        self._stop_event = threading.Event()

    def run(self):
        start_time = time.time()
        while not self._stop_event.is_set():
            if time.time() - start_time > 3:  # 3초 타임아웃
                break
            try:
                super().run()
            finally:
                self._stop_event.set()

    def stop(self):
        self._stop_event.set()

def make_logger(job_dir):
    # 로그 생성
    logger = logging.getLogger()

    # 로그의 출력 기준 설정
    logger.setLevel(logging.INFO)

    # log를 파일에 출력
    file_handler = logging.FileHandler(os.path.join(job_dir, 'log.txt'))
    logger.addHandler(file_handler)

    return logger

def move_with_timeout(client, x, y, z, duration):
    # 이동 명령 실행
    def move():
        client.moveByVelocityAsync(x, y, -z, 1).join()
    
    move_thread = StoppableThread(target=move)
    move_thread.start()
    
    # 설정된 타임아웃 후에 스레드 검사
    move_thread.join(timeout=duration)
    # time.sleep(3)
    if move_thread.is_alive():
        # 스레드가 여전히 실행 중이면, 필요한 경우 여기에 스레드 중단 로직을 추가
        move_thread.join()

def rotate_with_timeout(client, yaw_rate, radian, duration):
    # 회전 명령 실행
    def rotate():
        client.rotateToYawAsync(clacDuration(yaw_rate, radian)).join()
    
    rotate_thread = StoppableThread(target=rotate)
    rotate_thread.start()
    
    # 설정된 타임아웃 후에 스레드 검사
    rotate_thread.join(timeout=duration)
    # time.sleep(3)
    if rotate_thread.is_alive():
        # 스레드가 여전히 실행 중이면, 필요한 경우 여기에 스레드 중단 로직을 추가
        rotate_thread.join()

def convert_negative_radians_to_positive(radian):
    if radian >= 0:
        return radian  # If the angle is already positive, return as is.
    # Calculate the number of times 2π needs to be added
    n = math.ceil(abs(radian) / (2 * math.pi))
    return radian + 2 * math.pi * n

def clacDuration(yaw_rate, radian):
    # radian = convert_negative_radians_to_positive(radian)
    duration = (pi / radian) / (yaw_rate * pi / 180)
    return duration

def calcValues(qval, args):
    # action_vector를 PyTorch 텐서로 변환
    action_vector = torch.from_numpy(qval[0, :3])

    # Softmax 적용
    softmax = F.softmax(action_vector, dim=0).detach().cpu().numpy()
    action = np.random.choice(np.array([0, 1, 2]), p=softmax)
    radian = qval[0, 3]
    
    one_hot = np.zeros_like(action_vector.numpy())
    one_hot[action] = 1
    action_vector = one_hot * args["drone"].moving_unit
    action_vector = list(map(float, action_vector))

    radian = np.mod(radian + np.pi, 2 * np.pi) - np.pi
    
    return *action_vector, radian, action

def calcStatus(reward):
    status = "running"
    if abs(reward) != 2:
        if reward > 0:
            status = "work well"
        else:
            status = "work bad"
    return status

def calc_prev_curr_reward(prev_pcd, curr_pcd):
    len_ori_curr = len(curr_pcd.points)
    curr_pcd = prev_pcd + curr_pcd
    curr_pcd = curr_pcd.voxel_down_sample(voxel_size=0.05)
    len_prev = len(prev_pcd.points)
    len_curr = len(curr_pcd.points)
    
    if (len_curr - len_prev) / len_ori_curr < 0.05: 
        print("전에 만든 맵과 현재 만든 맵의 차이가 작으면 큰 음의 보상 (-10)")
        return -10
    else:
        print("전에 만든 맵과 현재 만든 맵의 차이가 크면 작은 음의 보상 (-1)")
        return -1
    
def calc_diff(mean1, mean2, std1, std2):
    mean_diff = np.abs(mean2 - mean1)
    std_diff = np.abs(std2 - std1)
    total_diff = np.sum(mean_diff**2 + std_diff**2)
    return total_diff

def calc_map_curr_reward(curr_pcd, map_infos, args):
    map_len, map_dist_mean, map_dist_std, map_density_mean, map_density_std = map_infos
    curr_len, curr_dist_mean, curr_dist_std, curr_density_mean, curr_density_std = extract_infos_from_pcd(curr_pcd)
    
    count_state = curr_len < map_len * 0.75
    dist_state = calc_diff(map_dist_mean, curr_dist_mean, map_dist_std, curr_dist_std) < args.reward_state_threshold
    density_state = calc_diff(map_density_mean, curr_density_mean, map_density_std, curr_density_std) < args.reward_state_threshold
    if count_state:
        print("map pcd와 current pcd의 차이가 크면 작은 음의 보상 (-1)")
        return -1
    elif dist_state: 
        print("map pcd와 current pcd의 분포 차이가 크면 작은 음의 보상 (-1)")
        return -1
    elif density_state:
        print("map pcd와 current pcd의 밀도 차이가 크면 작은 음의 보상 (-1)")
        return -1
    else:
        print("목표하던 바를 달성했으므로 큰 양의 보상 (+10)")
        return 10
    
def calcReward(map_infos, prev_pcd, curr_pcd, running_time, args):
    # 보상 초기값
    reward = 0.0

    # 드론이 충돌한 경우
    # if has_collided:
    #     reward -= 20.0  # 충돌 시 큰 음수 값으로 보상
    #     print("드론이 충돌한 경우")
    if running_time >= args.max_time:
        reward -= 10.0
        print("시간이 초과되어 큰 음의 보상 (-10)")

    # 전에 만든 맵과 현재 만든 맵의 차이에 따른 보상
    reward += calc_prev_curr_reward(prev_pcd, curr_pcd)

    # 전체 맵과 현재 만든 맵의 차이에 따른 보상
    mc_reward = calc_map_curr_reward(prev_pcd + curr_pcd, map_infos, args)
    if mc_reward == -1:   reward += mc_reward
    elif mc_reward == 10: reward = mc_reward

    print(f"total reward : {reward}")
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

def resetState(client, data_queue=None):
    print("\nreset client...")
    client.reset()
    if data_queue is not None: data_queue.put("reset")
    time.sleep(2)
    client.confirmConnection() # 클라이언트와 시뮬레이션서버간의 연결 확인, 연결이 안되어있으면 에러 발생
    client.enableApiControl(True) # 드론의 API 제어를 활성화, 활성화 해야 드론에 명령을 내릴 수 있음, 비활성화시 드론은 움직이지 않음.
    client.armDisarm(True) # 드론의 시동을 걸어줌, 가상 드론의 모터를 활성화하고 비행 준비 상태로 전환
    client.takeoffAsync().join() # 이륙, join()은 이륙 작업이 완료될때까지 기다리는 역할을 함
    client.hoverAsync().join() # 드론이 현재위치에서 정지비행상태로 전환하는 메서드, join은 완료할 때까지 기다리는 역할
    return client