import utils.setup_path as setup_path
import airsim

from collections import deque

import copy
import random
from datetime import datetime
import multiprocessing

import torch
import numpy as np
from math import pi

from tools.test import getAccuracy
from tools.models import MovePredictModel
from tools.vision import *
from tools.train_utils import *
from tools.live_visualization import live_visualization

from config.default import args

if __name__ == '__main__':
    job_dir = os.path.join("./run", datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss'))
    os.makedirs(job_dir, exist_ok=True)

    if args.live_visualization:
        data_queue = multiprocessing.Queue()  
        live_vis_process = multiprocessing.Process(target=live_visualization, args=(data_queue,))
        live_vis_process.start()

    device = args.device

    model1 = MovePredictModel(num_classes=args.num_classes, backbone="resnet101")
    model2 = copy.deepcopy(model1) 
    model2.load_state_dict(model1.state_dict()) 
    model1.to(device)
    model2.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=args.learning_rate)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    client.armDisarm(True)
    client.takeoffAsync().join()

    client.hoverAsync().join()

    best_acc = 0
    episode_step=0
    map_pcd = getMapPointCloud(client, args.voxel_size)

    replay = deque(maxlen=args.mem_size)
    losses = []

    for epoch in range(args.epochs):
        client.reset()
        if args["drone"].set_random_pose:
            setRandomPose(client, args)

        pcd_global = o3d.geometry.PointCloud()
        curr_pcd = mergePointClouds(pcd_global, getPointCloudByIntrinsic(client), client)
        pcd_global = copy.deepcopy(curr_pcd)
        if args.live_visualization:
            putDataIntoQueue(data_queue, pcd_global)

        rgb1 = getRGBImage(client)
        rgb1 = torch.from_numpy(rgb1).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)

        status = "running"
        running_time = 0
        while(status == "running"): 
            episode_step += 1
            qval = model1(rgb1)
            qval = qval.cpu().data.numpy()

            x, y, z, radian, action = calcValues(qval, args)
            
            move_start_time = time.time()
            client.moveToPositionAsync(x, y, z, args["drone"].default_velocity, 
                                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                    yaw_mode=airsim.YawMode(False, clacDuration(args["drone"].yaw_rate, radian))).join()
            running_time += time.time() - move_start_time

            rgb2 = getRGBImage(client)
            rgb2 = torch.from_numpy(rgb2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(device)

            curr_pcd = mergePointClouds(pcd_global, getPointCloudByIntrinsic(client), client)
            reward = calcReward(map_pcd, pcd_global, curr_pcd, client, running_time, args)
            pcd_global = copy.deepcopy(curr_pcd)
            if args.live_visualization:
                putDataIntoQueue(data_queue, pcd_global)

            done = True if reward > 0 else False
            exp =  (rgb1, rgb2, action, reward, done)
            replay.append(exp) 

            rgb1 = rgb2
            
            if len(replay) > args.batch_size:
                minibatch = random.sample(replay, args.batch_size)
                rgb1_batch   = torch.cat([i1 for (i1,i2,a,r,d) in minibatch], dim=0).to(device)
                rgb2_batch   = torch.cat([i2 for (i1,i2,a,r,d) in minibatch], dim=0).to(device)
                action_batch = torch.Tensor([a for (i1,i2,a,r,d) in minibatch]).to(device)
                reward_batch = torch.Tensor([r for (i1,i2,a,r,d) in minibatch]).to(device)
                done_batch   = torch.Tensor([d for (i1,i2,a,r,d) in minibatch]).to(device)

                Q1 = model1(rgb1_batch) 
                with torch.no_grad():
                    Q2 = model2(rgb2_batch) 
                
                Y = reward_batch + args.gamma * ((1 - done_batch) * (torch.max(Q2[:, :3],dim=1)[0] + Q2[:, 3])) #N
                X = Q1[:, :3].gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze() + Q1[:, 3]
                loss = criterion(X, Y.detach())
                print(f"epoch : {epoch}, loss : {loss.item()}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if episode_step % args.sync_freq == 0: 
                    model2.load_state_dict(model1.state_dict())
                    torch.save(model1.state_dict(), os.path.join(job_dir, f'{args.model_name}_{epoch}.pth'))

            status = calcStatus(reward)

        if epoch % args.eval_freq == 0 and epoch != 0:
            client.reset()
            acc = getAccuracy(model1, client, map_pcd, args)
            if best_acc < acc:
                best_acc = acc
                print("Save new best model...")
                torch.save(model1.state_dict(), os.path.join(job_dir, 'best_accurracy.pth'))

    if args.live_visualization:
        data_queue.put(None)
        live_vis_process.join()
    client.enableApiControl(False)
    losses = np.array(losses)
    np.save(os.path.join(job_dir, "losses.npy"), losses)