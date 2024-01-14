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
from tools.models import MovePredictModel, RGBDepthFusionNet
from tools.vision import *
from tools.train_utils import * 
from tools.live_visualization import live_visualization

from config.default import args

# main 함수
if __name__ == '__main__':
    # python 디랙토리를 만들고 현재 날짜와 시간을 포함하는 디렉토리 경로를 생성
    date = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    job_dir = os.path.join("./run", date)
    os.makedirs(job_dir, exist_ok=True)

    logger = make_logger(job_dir)

    data_queue = None
    if args.live_visualization: # 라이브 시각화를 위한 프로세스 생성
        data_queue = multiprocessing.Queue()  # 다중 프로세스간에 데이터를 공유 할 수 있는 큐 생성
        live_vis_process = multiprocessing.Process(target=live_visualization, args=(data_queue,)) # 별도의 프로세스로 실행
        live_vis_process.start() # 프로세스 시작

    device = args.device # cpu or gpu

    TDRS = TimeDecayRewardScheduler(**args["TDRS"])

    model1, model2 = create_models(args)
    criterion = torch.nn.MSELoss() # 손실함수 mse 사용
    optimizer = torch.optim.Adam(model1.parameters(), lr=args.learning_rate) # 최적화 알고리즘 아담사용

    client = airsim.MultirotorClient() # client 생성

    best_acc = 0  # 가장 높은 정확도
    total_episode_step = 0 # 에피소드의 현재 단계
    # map_voxel, map_infos = getMapVoxel(args.map_path) # 맵의 포인트 클라우드를 얻음
    map_pcd = getMapPointCloud(args) # 맵의 포인트 클라우드를 얻음
    min_x, max_x, min_y, max_y = getMinMaxXY()

    replay = deque(maxlen=args.mem_size) 
    losses = []

    for epoch in range(args.epochs):
        print_and_logging(logger, "\n\n" + "=" * 50)
        print_and_logging(logger, f"\nEpisode : {epoch}")
        client = resetState(client, data_queue) # 매 epoch 마다 client를 리셋
        initial_z = client.getMultirotorState().kinematics_estimated.position.z_val

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
            print_and_logging(logger, f"\nEpisode step : {episode_step}")

            position = getDronePositionTensor(client, min_x, max_x, min_y, max_y, device)

            qval = model1(rgb1, depth1, position) # 모델 1에 RGB 이미지를 입력하여 Q값을 

            action, move_or_rotate = calcValues(qval, client, logger, args) # Q값을 통해 드론의 위치와 행동을 계산
            move_start_time = time.time()   # 현재 시간을 측정, 드론이 움직이기 시작한 시간
            if action == 0:
                current_z = client.getMultirotorState().kinematics_estimated.position.z_val
                v = args["drone"].default_velocity
                x, y, _ = move_or_rotate
                z = (initial_z - current_z) * 0.5 
                move_with_timeout(client, x * v, y * v, z, 1)  # 3초 타임아웃
            else:
                rotate_with_timeout(client, move_or_rotate, 1)  # 3초 타임아웃
            
            client.simPause(True)
            running_time += time.time() - move_start_time   # 드론이 움직인 시간을 측정
            rgb2, depth2 = getImages(client)  # 드론의 카메라를 통해 RGB 이미지를 얻음
            rgb2 = torch.from_numpy(rgb2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)
            depth2 = torch.from_numpy(depth2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)
            curr_pcd = get_transformed_lidar_pc(client)
            decay_factor = TDRS.get_decay_factor(running_time)
            reward = calcReward(global_pcd, curr_pcd, running_time, args) 
            reward = reward * decay_factor if reward > 0 else reward
            print_and_logging(logger, f"total reward : {reward}")
            print_and_logging(logger, f"running_time : {running_time}")
            client.simPause(False)

            global_pcd = global_pcd + curr_pcd    # 현재 포인트 클라우드를 전체 클라우드에 복사
            global_pcd = global_pcd.voxel_down_sample(voxel_size=args.voxel_size)
            print_and_logging(logger, f"length of global_pcd.points : {len(global_pcd.points)}")
            if args.live_visualization: putDataIntoQueue(data_queue, curr_pcd)    # 3D 포인트 클라우드 데이터를 큐에 넣음

            done = calcDone(map_pcd, global_pcd)    # 양수의 보상이 나오면 해당 에피소드는 목표를 달성하여 종료
            print_and_logging(logger, f"is done : {done}")
            exp =  (rgb1, rgb2, depth1, depth2, position, action, reward, done) # 경험을 튜플로 생성, 경험은 RGB 이미지, 행동, 보상, 종료 여부로 구성
            if args.train_mode == "manual": saveEXP(exp, job_dir, f"episode{epoch}_step{episode_step}")
            replay.append(exp) # 경험을 메모리에 저장. replay는 경험을 저장하는 리스트.

            rgb1 = copy.deepcopy(rgb2) # 다음 상태에서 쓸 이전 상태값을 보존하기 위한 깊은 복사
            depth1 = copy.deepcopy(depth2) # 다음 상태에서 쓸 이전 상태값을 보존하기 위한 깊은 복사

            if len(replay) > args.batch_size:   # 메모리에 저장된 경험의 수가 배치 사이즈보다 크면
                minibatch = random.sample(replay, args.batch_size)
                rgb1_batch = torch.cat([i1 for (i1, i2, d1, d2, p, a, rw, d) in minibatch], dim=0).to(device)
                rgb2_batch = torch.cat([i2 for (i1, i2, d1, d2, p, a, rw, d) in minibatch], dim=0).to(device)
                depth1_batch = torch.cat([d1 for (i1, i2, d1, d2, p, a, rw, d) in minibatch], dim=0).to(device)
                depth2_batch = torch.cat([d2 for (i1, i2, d1, d2, p, a, rw, d) in minibatch], dim=0).to(device)
                position_batch = torch.cat([p for (i1, i2, d1, d2, p, a, rw, d) in minibatch], dim=0).to(device)
                action_batch = torch.Tensor([a for (i1, i2, d1, d2, p, a, rw, d) in minibatch]).to(device)
                reward_batch = torch.Tensor([rw for (i1, i2, d1, d2, p, a, rw, d) in minibatch]).to(device)
                done_batch = torch.Tensor([d for (i1, i2, d1, d2, p, a, rw, d) in minibatch]).to(device)

                Q1 = model1(rgb1_batch, depth1_batch, position_batch)
                with torch.no_grad():
                    Q2 = model2(rgb2_batch, depth2_batch, position_batch)

                # 이동 액션에 대한 Q 값
                Q1 = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

                # 타깃 Q 값 계산
                Y = reward_batch + args.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])

                # 손실 계산: 이동과 회전 액션에 대한 손실을 각각 계산
                loss_move = criterion(Q1, Y.detach())
                print_and_logging(logger, f"loss : {loss_move.item()}")

                optimizer.zero_grad()
                loss_move.backward()
                optimizer.step()
                
                if total_episode_step % args.sync_freq == 0: # args.sync_freq는 동기화 주기, 업데이트 및 저장을 수행하는 빈도
                    model2.load_state_dict(model1.state_dict()) 
                    torch.save(model2.state_dict(), os.path.join(job_dir, f'model_step{total_episode_step}.pth')) 

            status = calcStatus(reward, done) # 보상을 기반으로 현재 에피소드 상태를 계산.
            print_and_logging(logger, f"status : {status}")
        
        if epoch % args.eval_freq == 0 and epoch != 0: # 일정 주기마다 모델을 평가
            o3d.io.write_point_cloud(os.path.join(job_dir, f'global_pcd_epoch{epoch}.ply'), global_pcd)
            acc = getAccuracy(model2, client, map_pcd, logger, args)    # 모델의 정확도를 계산
            print_and_logging(logger, f"Accuracy : {acc} (best : {best_acc})")
            if best_acc < acc: # 정확도가 높아지면
                best_acc = acc # 정확도를 갱신
                print_and_logging(logger, "Save new best model...")
                torch.save(model2.state_dict(), os.path.join(job_dir, 'best_accurracy.pth')) 

    if args.live_visualization: # 라이브 시각화 프로세스가 동작중이면
        live_vis_process.join() # 프로세스가 종료될 때까지 기다림
    client.enableApiControl(False)  # 드론의 API 제어를 비활성화
    losses = np.array(losses) # 리스트를 numpy 배열로 변환
    np.save(os.path.join(job_dir, "losses.npy"), losses) # 손실함수를 저장