import utils.setup_path as setup_path
import airsim

from math import pi
import numpy as np
import torch
import torch.nn.functional as F
import random
import time

from tools.vision import *
from tools.train_utils import *
from tools.models import RGBDepthFusionNet

import threading

import logging

class TimeDecayRewardScheduler():
    def __init__(self, decay_factor, min_decay_factor, every_second):
        self.decay_factor = decay_factor
        self.min_decay_factor = min_decay_factor
        self.es = every_second
    
    def get_decay_factor(self, running_time):
        t = running_time // self.es
        result = self.decay_factor ** t
        if result < self.min_decay_factor:
            return self.min_decay_factor
        return result

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

def create_models(args):
    model1 = RGBDepthFusionNet(num_classes=args.num_classes, backbone=args.backbone_name) # 모델 생성
    if args.checkpoint is not None: model1.load_state_dict(torch.load(args.checkpoint))
    model2 = copy.deepcopy(model1) # 모델 생성 22
    model2.load_state_dict(model1.state_dict())  # 모델 2에 모델 1의 가중치와 매개변수를 복사
    model1.to(args.device) # 모델 1을 cpu or gpu에 할당
    model2.to(args.device) # 모델 2를 cpu or gpu에 할당
    return model1, model2

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
        client.moveByVelocityAsync(x, y, z, 1).join()
    
    move_thread = StoppableThread(target=move)
    move_thread.start()
    
    # 설정된 타임아웃 후에 스레드 검사
    move_thread.join(timeout=duration)
    # time.sleep(3)
    if move_thread.is_alive():
        # 스레드가 여전히 실행 중이면, 필요한 경우 여기에 스레드 중단 로직을 추가
        move_thread.join()

def rotate_with_timeout(client, degree, duration):
    # 회전 명령 실행
    def rotate():
        client.rotateToYawAsync(degree).join()
    
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

def calcDuration(yaw_rate, radian):
    # radian = convert_negative_radians_to_positive(radian)
    duration = (pi / radian) / (yaw_rate * pi / 180)
    return duration

def getClosestDegree(client):
    state = client.getMultirotorState()
    orientation = state.kinematics_estimated.orientation
    degree = math.degrees(airsim.to_eularian_angles(orientation)[2])

    if degree < 0:
        degree = 360 + degree

    angles = [0, 90, 180, 270, 360]
    closest = min(angles, key=lambda x: abs(x - degree))
    return closest

def calcForwardDirection(client):
    degree = getClosestDegree(client)
    forward_vector = airsim.Vector3r(math.cos(math.radians(degree)), math.sin(math.radians(degree)), 0)
    return [forward_vector.x_val, forward_vector.y_val, forward_vector.z_val]

def calcDegree(client, action):
    degree = getClosestDegree(client)
    if   action == 1: degree -= 90
    elif action == 2: degree += 90
    return degree

def calcValues(qval, client, args):
    softmax = F.softmax(torch.from_numpy(qval[0, :3]), dim=0).detach().cpu().numpy()
    action = np.random.choice(np.array([0, 1, 2]), p=softmax)
    if args.train_mode == "manual": 
        while 1:
            try:
                action = int(input("select action (0, 1, 2) : "))
            except:
                print("wrong input...")
                continue
            if action == 0 or action == 1 or action == 2: break
            else: print("wrong input...")
    if action == 0: return action, calcForwardDirection(client)
    else:           return action, calcDegree(client, action)

def calcStatus(reward, done):
    status = "running"
    if reward > 0 and done:
        status = "work well"
    elif reward < 0:
        status = "work bad"
    return status

def calcDone(map_pcd, curr_pcd):
    done = len(curr_pcd.points) >= len(map_pcd.points) * 0.7
    return done

def calcReward(prev_pcd, curr_pcd, running_time, args):
    if running_time >= args.max_time:
        reward = -10.0
        return reward
    
    len_ori_curr = len(curr_pcd.points)
    len_prev = len(prev_pcd.points)
    curr_pcd = prev_pcd + curr_pcd
    curr_pcd = curr_pcd.voxel_down_sample(voxel_size=0.05)
    len_curr = len(curr_pcd.points)
    reward = abs(len_curr - len_prev) / len_ori_curr

    return reward

def getMinMaxXY():
   min_x, max_x = -3.3802177906036377, 16.81801986694336
   min_y, max_y = -1.6409467458724976, 19.18399429321289
   return min_x, max_x, min_y, max_y

def normalize_position(x, y, min_x, max_x, min_y, max_y):
    normalized_x = (x - min_x) / (max_x - min_x)
    normalized_y = (y - min_y) / (max_y - min_y)
    return normalized_x, normalized_y

def normalize_yaw(yaw):
    # Yaw 값을 0도에서 360도 범위로 조정 (필요한 경우)
    if yaw < 0:
        yaw += 360

    # Yaw 값을 0과 1 사이로 정규화
    normalized_yaw = yaw / 360
    return normalized_yaw

def getDronePositionTensor(client, min_x, max_x, min_y, max_y, device):
    # 드론 상태 가져오기
    drone_state = client.getMultirotorState()

    # GPS 좌표에서 x, y 위치 추출
    local_position = drone_state.kinematics_estimated.position
    x = local_position.x_val
    y = local_position.y_val

    # 위치 정규화
    x, y = normalize_position(x, y, min_x, max_x, min_y, max_y)

    # 쿼터니언에서 Yaw(방향) 각도 추출
    orientation = drone_state.kinematics_estimated.orientation
    yaw = normalize_yaw(math.degrees(airsim.to_eularian_angles(orientation)[2]))

    position_tensor = torch.FloatTensor([x, y, yaw]).to(device)
    return position_tensor.unsqueeze(dim=0)

def saveEXP(exp, job_dir, name_tag):
    rgb1, rgb2, depth1, depth2, position, action, reward, done = exp
    
    rgb1 = rgb1.cpu().detach().data.numpy()
    rgb2 = rgb2.cpu().detach().data.numpy()
    depth1 = depth1.cpu().detach().data.numpy()
    depth2 = depth2.cpu().detach().data.numpy()

    save_path = os.path.join(job_dir, "exp")
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, f"rgb1_{name_tag}.npy"), rgb1)
    np.save(os.path.join(save_path, f"rgb2_{name_tag}.npy"), rgb2)
    np.save(os.path.join(save_path, f"depth1_{name_tag}.npy"), depth1)
    np.save(os.path.join(save_path, f"depth2_{name_tag}.npy"), depth2)

    with open(os.path.join(save_path, f"values_{name_tag}.txt"), "w") as f:
        save_str = f"{position[0, 0]} {position[0, 1]} {position[0, 2]} {action} {reward} {int(done)}"
        f.write(save_str)
    
    with open(os.path.join(job_dir, "name_tags.txt"), 'a') as f:
        f.write(name_tag + '\n')

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