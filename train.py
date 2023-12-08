import utils.setup_path as setup_path
import airsim

from collections import deque

import copy
import random
from datetime import datetime
import multiprocessing
import logging

import torch
import numpy as np

from tools.test import getAccuracy
from tools.models import MovePredictModel, RGBDepthFusionNet
from tools.vision import *
from tools.train_utils import * 
from tools.live_visualization import live_visualization

from config.default import args

# main 함수
if __name__ == '__main__':
    # python 디랙토리를 만들고 현재 날짜와 시간을 포함하는 디렉토리 경로를 생성
    job_dir = os.path.join("./run", datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss'))
    os.makedirs(job_dir, exist_ok=True)

    logger = make_logger(job_dir)

    data_queue = None
    if args.live_visualization: # 라이브 시각화를 위한 프로세스 생성
        data_queue = multiprocessing.Queue()  # 다중 프로세스간에 데이터를 공유 할 수 있는 큐 생성
        live_vis_process = multiprocessing.Process(target=live_visualization, args=(data_queue,)) # 별도의 프로세스로 실행
        live_vis_process.start() # 프로세스 시작

    device = args.device # cpu or gpu

    model1 = RGBDepthFusionNet(num_classes=args.num_classes, backbone=args.backbone_name) # 모델 생성
    model2 = copy.deepcopy(model1) # 모델 생성 22
    model2.load_state_dict(model1.state_dict())  # 모델 2에 모델 1의 가중치와 매개변수를 복사
    model1.to(device) # 모델 1을 cpu or gpu에 할당
    model2.to(device) # 모델 2를 cpu or gpu에 할당

    criterion = torch.nn.MSELoss() # 손실함수 mse 사용
    optimizer = torch.optim.Adam(model1.parameters(), lr=args.learning_rate) # 최적화 알고리즘 아담사용

    client = airsim.MultirotorClient() # client 생성

    best_acc = 0  # 가장 높은 정확도
    total_episode_step = 0 # 에피소드의 현재 단계
    # map_voxel, map_infos = getMapVoxel(args.map_path) # 맵의 포인트 클라우드를 얻음
    map_pcd = getMapPointCloud(args.map_path) # 맵의 포인트 클라우드를 얻음
    map_infos = extract_infos_from_pcd(map_pcd)

    replay = deque(maxlen=args.mem_size) 
    losses = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpisode : {epoch}")
        client = resetState(client, data_queue) # 매 epoch 마다 client를 리셋
        if args["drone"].set_random_pose: setRandomPose(client, args) # 드론의 위치를 랜덤으로 설정

        global_pcd = o3d.geometry.PointCloud() # 빈 3D 포인트 클라우드 생성, 이후에 드론의 위치를 기준으로 포인트 클라우드를 추가하여 채움

        client.simPause(True)
        rgb1, depth1 = getImages(client) # 드론의 카메라를 통해 RGB 이미지를 얻음
        rgb1 = torch.from_numpy(rgb1).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)
        depth1 = torch.from_numpy(depth1).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)
        curr_pcd = get_transformed_lidar_pc(client) # 현재 포인트 클라우드를 전체 클라우드에 병합, 현재시점의 데이터를 전체 데이터에 누적
        client.simPause(False)

        global_pcd = global_pcd + curr_pcd # 현재 포인트 클라우드를 전체 클라우드에 복사
        if args.live_visualization: putDataIntoQueue(data_queue, curr_pcd) # 3D 포인트 클라우드 데이터를 큐에 넣음

        status = "running"  # 상태를 running으로 설정
        running_time = 0    # 드론이 움직인 시간
        episode_step = 0 # 에피소드의 현재 단계
        while(status == "running"):     # 상태가 running일때 반복
            episode_step += 1   # 에피소드의 현재 단계를 1 증가
            total_episode_step += 1
            logger.info(f"Episode step : {episode_step}")
            qval = model1(rgb1, depth1)     # 모델 1에 RGB 이미지를 입력하여 Q값을 
            qval = qval.cpu().data.numpy()  # Q값을 numpy 배열로 변환

            x, y, z, radian, action = calcValues(qval, args) # Q값을 통해 드론의 위치와 행동을 계산
            # print("x, y, z, radian, action :", x, y, z, radian, action)
            move_start_time = time.time()   # 현재 시간을 측정, 드론이 움직이기 시작한 시간
            move_with_timeout(client, x, y, z, 3)  # 3초 타임아웃
            rotate_with_timeout(client, args["drone"].yaw_rate, radian, 3)  # 3초 타임아웃
            
            client.simPause(True)
            running_time += time.time() - move_start_time   # 드론이 움직인 시간을 측정
            rgb2, depth2 = getImages(client)  # 드론의 카메라를 통해 RGB 이미지를 얻음
            rgb2 = torch.from_numpy(rgb2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)
            depth2 = torch.from_numpy(depth2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)
            curr_pcd = get_transformed_lidar_pc(client)   # 현재 포인트 클라우드를 전체 클라우드에 병합, 현재시점의 데이터를 전체 데이터에 누적
            reward = calcReward(map_infos, global_pcd, curr_pcd, running_time, args)   # 보상을 계산, 맵의 포인트 클라우드, 전체 포인트 클라우드, 현재 포인트 클라우드, 드론이 움직인 시간, args를 인자로 전달
            logger.info(f"total reward : {reward}")
            client.simPause(False)

            global_pcd = global_pcd + curr_pcd    # 현재 포인트 클라우드를 전체 클라우드에 복사
            if args.live_visualization: putDataIntoQueue(data_queue, curr_pcd)    # 3D 포인트 클라우드 데이터를 큐에 넣음

            done = True if reward > 0 else False    # 양수의 보상이 나오면 해당 에피소드는 목표를 달성하여 종료
            exp =  (rgb1, rgb2, depth1, depth2, action, radian, reward, done) # 경험을 튜플로 생성, 경험은 RGB 이미지, 행동, 보상, 종료 여부로 구성
            replay.append(exp) # 경험을 메모리에 저장. replay는 경험을 저장하는 리스트.

            rgb1 = copy.deepcopy(rgb2) # 다음 상태에서 쓸 이전 상태값을 보존하기 위한 깊은 복사
            depth1 = copy.deepcopy(depth2) # 다음 상태에서 쓸 이전 상태값을 보존하기 위한 깊은 복사

            if len(replay) > args.batch_size:   # 메모리에 저장된 경험의 수가 배치 사이즈보다 크면
                # minibatch = random.sample(replay, args.batch_size)  # 메모리에서 배치 사이즈만큼 경험을 랜덤하게 추출
                # rgb1_batch   = torch.cat([i1 for (i1,i2,a,r,d) in minibatch], dim=0).to(device) # 경험에서 RGB 이미지를 추출하여 배치로 만듬
                # rgb2_batch   = torch.cat([i2 for (i1,i2,a,r,d) in minibatch], dim=0).to(device) # 경험에서 RGB 이미지를 추출하여 배치로 만듬
                # action_batch = torch.Tensor([a for (i1,i2,a,r,d) in minibatch]).to(device) # 경험에서 행동을 추출하여 배치로 만듬
                # reward_batch = torch.Tensor([r for (i1,i2,a,r,d) in minibatch]).to(device) # 경험에서 보상을 추출하여 배치로 만듬
                # done_batch   = torch.Tensor([d for (i1,i2,a,r,d) in minibatch]).to(device) # 경험에서 종료 여부를 추출하여 배치로 만듬

                # Q1 = model1(rgb1_batch) # 모델 1에 RGB 이미지를 입력하여 Q값을 얻음
                # with torch.no_grad(): # 기울기를 계산하지 않음, 모든 연산이 기록되지않아 역전파가 수행되지못함 -> 순전파로만 Q값을 계산
                #     Q2 = model2(rgb2_batch) # 모델 2에 RGB 이미지를 입력하여 Q값을 얻음

                # Y = reward_batch + args.gamma * ((1 - done_batch) * (torch.max(Q2[:, :3],dim=1)[0] + Q2[:, 3]))
                # X = Q1[:, :3].gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze() + Q1[:, 3] 

                # loss = criterion(X, Y.detach()) # 손실함수 계산, criterion은 MSE 손실함수
                # logger.info(f"loss : {loss.item()}") # 에포크와 손실함수 출력
                # print(f"total loss : {loss}")
                # optimizer.zero_grad()   # 기울기 초기화
                # loss.backward() # 손실함수 역전파
                # optimizer.step() # 최적화 알고리즘을 통해 가중치와 매개변수를 업데이트

                minibatch = random.sample(replay, args.batch_size)
                rgb1_batch = torch.cat([i1 for (i1, i2, d1, d2, a, rd, rw, d) in minibatch], dim=0).to(device)
                rgb2_batch = torch.cat([i2 for (i1, i2, d1, d2, a, rd, rw, d) in minibatch], dim=0).to(device)
                depth1_batch = torch.cat([d1 for (i1, i2, d1, d2, a, rd, rw, d) in minibatch], dim=0).to(device)
                depth2_batch = torch.cat([d2 for (i1, i2, d1, d2, a, rd, rw, d) in minibatch], dim=0).to(device)
                action_batch = torch.Tensor([a for (i1, i2, d1, d2, a, rd, rw, d) in minibatch]).to(device)
                radian_batch = torch.Tensor([rd for (i1, i2, d1, d2, a, rd, rw, d) in minibatch]).to(device)
                reward_batch = torch.Tensor([rw for (i1, i2, d1, d2, a, rd, rw, d) in minibatch]).to(device)
                done_batch = torch.Tensor([d for (i1, i2, d1, d2, a, rd, rw, d) in minibatch]).to(device)

                Q1 = model1(rgb1_batch, depth1_batch)
                with torch.no_grad():
                    Q2 = model2(rgb2_batch, depth2_batch)

                # 이동 액션에 대한 Q 값
                Q1_move = Q1[:, :3].gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                # 회전 액션에 대한 Q 값
                Q1_rotate = Q1[:, 3]

                # 타깃 Q 값 계산
                Y_move = reward_batch + args.gamma * ((1 - done_batch) * torch.max(Q2[:, :3], dim=1)[0])
                Y_rotate = reward_batch + args.gamma * ((1 - done_batch) * Q2[:, 3])

                # 손실 계산: 이동과 회전 액션에 대한 손실을 각각 계산
                loss_move = criterion(Q1_move, Y_move.detach())
                loss_rotate = criterion(Q1_rotate, Y_rotate.detach())
                total_loss = loss_move + loss_rotate

                logger.info(f"loss : {total_loss.item()}")
                # print(f"total loss : {total_loss}")
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if total_episode_step % args.sync_freq == 0: # args.sync_freq는 동기화 주기, 업데이트 및 저장을 수행하는 빈도
                    model2.load_state_dict(model1.state_dict()) # 현재 학습중인 모델1의 가중치와 편향을 모델2에 복사. 타깃네트워크와 현재 네트워크를 동일하게 만들기 위함
                    torch.save(model1.state_dict(), os.path.join(job_dir, f'model_step{total_episode_step}.pth')) # 모델1의 가중치와 편향을 저장

            status = calcStatus(reward) # 보상을 기반으로 현재 에피소드 상태를 계산.

        if epoch % args.eval_freq == 0 and epoch != 0: # 일정 주기마다 모델을 평가
            o3d.io.write_point_cloud(os.path.join(job_dir, f'global_pcd_epoch{epoch}.ply'), global_pcd)
            acc = getAccuracy(model1, client, map_infos, logger, args)    # 모델의 정확도를 계산
            logger.info(f"Accuracy : {acc} (best : {best_acc})")
            if best_acc < acc: # 정확도가 높아지면
                best_acc = acc # 정확도를 갱신
                print("Save new best model...") # 새로운 최고 정확도를 출력
                torch.save(model1.state_dict(), os.path.join(job_dir, 'best_accurracy.pth')) # 모델1의 가중치와 편향을 저장

    if args.live_visualization: # 라이브 시각화 프로세스가 동작중이면
        live_vis_process.join() # 프로세스가 종료될 때까지 기다림
    client.enableApiControl(False)  # 드론의 API 제어를 비활성화
    losses = np.array(losses) # 리스트를 numpy 배열로 변환
    np.save(os.path.join(job_dir, "losses.npy"), losses) # 손실함수를 저장