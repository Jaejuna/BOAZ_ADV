import utils.setup_path as setup_path
import airsim

from collections import deque

import copy
import random
from datetime import datetime
import multiprocessing

import torch
import numpy as np

from tools.test import getAccuracy
from tools.models import MovePredictModel
from tools.vision import *
from tools.train_utils import * 
from tools.live_visualization import live_visualization

from config.default import args

# main 함수
if __name__ == '__main__':
    # python 디랙토리를 만들고 현재 날짜와 시간을 포함하는 디렉토리 경로를 생성
    job_dir = os.path.join("./run", datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss'))
    os.makedirs(job_dir, exist_ok=True)

    if args.live_visualization: # 라이브 시각화를 위한 프로세스 생성
        data_queue = multiprocessing.Queue()  # 다중 프로세스간에 데이터를 공유 할 수 있는 큐 생성
        live_vis_process = multiprocessing.Process(target=live_visualization, args=(data_queue,)) # 별도의 프로세스로 실행
        live_vis_process.start() # 프로세스 시작

    device = args.device # cpu or gpu

    model1 = MovePredictModel(num_classes=args.num_classes, backbone=args.backbone_name) # 모델 생성
    model2 = copy.deepcopy(model1) # 모델 생성 22
    model2.load_state_dict(model1.state_dict())  # 모델 2에 모델 1의 가중치와 매개변수를 복사
    model1.to(device) # 모델 1을 cpu or gpu에 할당
    model2.to(device) # 모델 2를 cpu or gpu에 할당

    criterion = torch.nn.MSELoss() # 손실함수 mse 사용
    optimizer = torch.optim.Adam(model1.parameters(), lr=args.learning_rate) # 최적화 알고리즘 아담사용

    client = connectToClient() # client 생성
    client = droneReadyState(client) # 드론을 준비 완료 상태로 만듦

    best_acc = 0  # 가장 높은 정확도
    episode_step=0 # 에피소드의 현재 단계
    map_voxel, map_info = getMapVoxel(args.map_path) # 맵의 포인트 클라우드를 얻음

    replay = deque(maxlen=args.mem_size) 
    losses = []

    for epoch in range(args.epochs):
        client = resetState(client) # 매 epoch 마다 client를 리셋
        if args["drone"].set_random_pose:
            setRandomPose(client, args) # 드론의 위치를 랜덤으로 설정

        global_pcd = o3d.geometry.PointCloud() # 빈 3D 포인트 클라우드 생성, 이후에 드론의 위치를 기준으로 포인트 클라우드를 추가하여 채움

        client.simPause(True)
        rgb1 = getRGBImage(client) # 드론의 카메라를 통해 RGB 이미지를 얻음
        rgb1 = torch.from_numpy(rgb1).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)   # RGB 이미지를 텐서로 변환 
                                                                                                # unsqueeze(dim=0)은 차원을 추가. 여기서는 배치차원을 추가하여 입력을 한번에 전달하기 위함
                                                                                                # permute(0, 3, 1, 2) 이미지 데이터의 순서를 배치, 높이, 너비, 채널 순서로 변경
        curr_pcd = get_transformed_lidar_pc(client) # 현재 포인트 클라우드를 전체 클라우드에 병합, 현재시점의 데이터를 전체 데이터에 누적
        client.simPause(False)
        global_pcd = global_pcd + curr_pcd # 현재 포인트 클라우드를 전체 클라우드에 복사
        if args.live_visualization:
            putDataIntoQueue(data_queue, global_pcd) # 3D 포인트 클라우드 데이터를 큐에 넣음

        status = "running"  # 상태를 running으로 설정
        running_time = 0    # 드론이 움직인 시간
        while(status == "running"):     # 상태가 running일때 반복
            episode_step += 1   # 에피소드의 현재 단계를 1 증가
            qval = model1(rgb1)     # 모델 1에 RGB 이미지를 입력하여 Q값을 얻음
            qval = qval.cpu().data.numpy()  # Q값을 numpy 배열로 변환

            x, y, z, radian, action = calcValues(qval, args) # Q값을 통해 드론의 위치와 행동을 계산

            move_start_time = time.time()   # 현재 시간을 측정, 드론이 움직이기 시작한 시간
            client.moveToPositionAsync(x, y, z, args["drone"].default_velocity).join() # 드론을 x, y, z 위치로 이동시킴, velocity는 드론의 속도
            client.rotateToYawAsync(clacDuration(args["drone"].yaw_rate, radian)).join() 

            client.simPause(True)
            running_time += time.time() - move_start_time   # 드론이 움직인 시간을 측정
            rgb2 = getRGBImage(client)  # 드론의 카메라를 통해 RGB 이미지를 얻음
            rgb2 = torch.from_numpy(rgb2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)  # RGB 이미지를 텐서로 변환
                                                                                                    # unsqueeze(dim=0)은 차원을 추가. 여기서는 배치차원을 추가하여 입력을 한번에 전달하기 위함
                                                                                                    # permute(0, 3, 1, 2) 이미지 데이터의 순서를 배치, 높이, 너비, 채널 순서로 변경
            curr_pcd = get_transformed_lidar_pc(client)   # 현재 포인트 클라우드를 전체 클라우드에 병합, 현재시점의 데이터를 전체 데이터에 누적
            reward = calcReward(map_voxel, map_info, global_pcd, curr_pcd, client, running_time, args)   # 보상을 계산, 맵의 포인트 클라우드, 전체 포인트 클라우드, 현재 포인트 클라우드, 드론이 움직인 시간, args를 인자로 전달
            client.simPause(False)
            global_pcd = global_pcd + curr_pcd    # 현재 포인트 클라우드를 전체 클라우드에 복사
            if args.live_visualization:
                putDataIntoQueue(data_queue, global_pcd)    # 3D 포인트 클라우드 데이터를 큐에 넣음

            done = True if reward > 0 else False    # 양수의 보상이 나오면 해당 에피소드는 목표를 달성하여 종료
            exp =  (rgb1, rgb2, action, reward, done) # 경험을 튜플로 생성, 경험은 RGB 이미지, 행동, 보상, 종료 여부로 구성
            replay.append(exp) # 경험을 메모리에 저장. replay는 경험을 저장하는 리스트.

            rgb1 = copy.deepcopy(rgb2) # 다음 상태에서 쓸 이전 상태값을 보존하기 위한 깊은 복사

            if len(replay) > args.batch_size:   # 메모리에 저장된 경험의 수가 배치 사이즈보다 크면
                minibatch = random.sample(replay, args.batch_size)  # 메모리에서 배치 사이즈만큼 경험을 랜덤하게 추출
                rgb1_batch   = torch.cat([i1 for (i1,i2,a,r,d) in minibatch], dim=0).to(device) # 경험에서 RGB 이미지를 추출하여 배치로 만듬
                rgb2_batch   = torch.cat([i2 for (i1,i2,a,r,d) in minibatch], dim=0).to(device) # 경험에서 RGB 이미지를 추출하여 배치로 만듬
                action_batch = torch.Tensor([a for (i1,i2,a,r,d) in minibatch]).to(device) # 경험에서 행동을 추출하여 배치로 만듬
                reward_batch = torch.Tensor([r for (i1,i2,a,r,d) in minibatch]).to(device) # 경험에서 보상을 추출하여 배치로 만듬
                done_batch   = torch.Tensor([d for (i1,i2,a,r,d) in minibatch]).to(device) # 경험에서 종료 여부를 추출하여 배치로 만듬

                Q1 = model1(rgb1_batch) # 모델 1에 RGB 이미지를 입력하여 Q값을 얻음
                with torch.no_grad(): # 기울기를 계산하지 않음, 모든 연산이 기록되지않아 역전파가 수행되지못함 -> 순전파로만 Q값을 계산
                    Q2 = model2(rgb2_batch) # 모델 2에 RGB 이미지를 입력하여 Q값을 얻음

                Y = reward_batch + args.gamma * ((1 - done_batch) * (torch.max(Q2[:, :3],dim=1)[0] + Q2[:, 3])) # 타깃 Q값 계산
                                                                                                                # args.gamma는 감가율
                                                                                                                # done_batch는 종료 여부
                                                                                                                # (1 - done_batch)는 종료가 되지 않았을 때, 이때만 감가율을 곱함
                                                                                                                # torch.max(Q2[:, :3],dim=1)[0]는 Q2의 0~2번째 액션 중 가장 큰 값
                                                                                                                # Q2[:, 3]는 다음상태에 대한 특정 액션, 여기서는 Q2의 3번째 액션
                X = Q1[:, :3].gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze() + Q1[:, 3] # 현재 Q값 계산
                                                                                                            # Q1[:, :3]는 Q1의 0~2번째 액션
                                                                                                            # action_batch는 선택한 액션의 인덱스를 나타냄
                                                                                                            # gather 함수를 사용하여 선택한 액션에 대한 Q값을 가져옴
                                                                                                            # squeeze 함수를 사용하여 차원을 줄임
                                                                                                            # Q1[:, 3]는 다음상태에 대한 특정 액션, 여기서는 Q1의 3번째 액션
                loss = criterion(X, Y.detach()) # 손실함수 계산, criterion은 MSE 손실함수
                print(f"epoch : {epoch}, loss : {loss.item()}") # 에포크와 손실함수 출력
                optimizer.zero_grad()   # 기울기 초기화
                loss.backward() # 손실함수 역전파
                optimizer.step() # 최적화 알고리즘을 통해 가중치와 매개변수를 업데이트
                
                if episode_step % args.sync_freq == 0: # 에피소드의 현재 단계수, args.sync_freq는 동기화 주기, 업데이트 및 저장을 수행하는 빈도
                    model2.load_state_dict(model1.state_dict()) # 현재 학습중인 모델1의 가중치와 편향을 모델2에 복사. 타깃네트워크와 현재 네트워크를 동일하게 만들기 위함
                    torch.save(model1.state_dict(), os.path.join(job_dir, f'{args.model_name}_{epoch}.pth')) # 모델1의 가중치와 편향을 저장

            status = calcStatus(reward) # 보상을 기반으로 현재 에피소드 상태를 계산.

        if epoch % args.eval_freq == 0 and epoch != 0: # 일정 주기마다 모델을 평가
            o3d.io.write_point_cloud(os.path.join(job_dir, f'global_pcd_epoch{epoch}.ply'), global_pcd)
            acc = getAccuracy(model1, client, map_voxel, map_info, args)    # 모델의 정확도를 계산
            if best_acc < acc: # 정확도가 높아지면
                best_acc = acc # 정확도를 갱신
                print("Save new best model...") # 새로운 최고 정확도를 출력
                torch.save(model1.state_dict(), os.path.join(job_dir, 'best_accurracy.pth')) # 모델1의 가중치와 편향을 저장

    if args.live_visualization: # 라이브 시각화 프로세스가 동작중이면
        data_queue.put(None)    # 큐에 None을 넣어서 종료
        live_vis_process.join() # 프로세스가 종료될 때까지 기다림
    client.enableApiControl(False)  # 드론의 API 제어를 비활성화
    losses = np.array(losses) # 리스트를 numpy 배열로 변환
    np.save(os.path.join(job_dir, "losses.npy"), losses) # 손실함수를 저장