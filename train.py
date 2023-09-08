from collections import deque

import copy
import random
from IPython.display import clear_output

from datetime import datetime

import torch
import numpy as np
from math import pi

import setup_path
import airsim

from loss import VoxelMSELoss
from model import MovePredictModel
from vision import *
from config import args
from train_utils import *

job_dir = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')

device = args.device

model1 = MovePredictModel(num_classes=args.num_classes, backbone="resnet101")
model2 = copy.deepcopy(model1) 
model2.load_state_dict(model1.state_dict()) 
model1.to(device)
model2.to(device)

criterion = VoxelMSELoss((0, 0, 0))
optimizer = torch.optim.Adam(model1.parameters(), lr=args.learning_rate)

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

client.hoverAsync().join()

best_acc = 0
episode_step=0
map_pcd = getMapPointCloud(client)

replay = deque(maxlen=args.mem_size)
losses = []

for epoch in range(args.epochs):
    client.reset()

    pcd_global = o3d.geometry.PointCloud()
    pcd_global = merge_point_clouds(pcd_global, getPointCloud(client), client)

    rgb1, depth1 = getImages(client)
    rgb1 = torch.from_numpy(rgb1).to(device)

    status = "running"
    start_time = time.time()
    while(status == "running"): 
        episode_step += 1
        qval = model1(rgb1)
        qval = qval.data.numpy()

        x, y, z, radian = calcValues(qval, args.epsilon)
        action = (x, y, z, radian)
        
        client.moveToPositionAsync(x, y, z, args["drone"].defualt_velocity, 
                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                   yaw_mode=airsim.YawMode(False, clacDuration(args["drone"].yaw_rate, radian))).join()

        rgb2, depth2 = getImages(client)
        rgb2 = torch.from_numpy(rgb2).to(device)

        curr_pcd = merge_point_clouds(pcd_global, getPointCloud(client), client)
        reward = calcReward(map_pcd, pcd_global, curr_pcd, client)
        pcd_global = copy.deepcopy(curr_pcd)

        done = True if reward > 0 else False
        exp =  (rgb1, rgb2, action, reward, done, pcd_to_voxel_tensor(pcd_global))
        replay.append(exp) 

        rgb1 = rgb2
        
        if len(replay) > args.batch_size:
            minibatch = random.sample(replay, args.batch_size)
            rgb1_batch   = torch.cat([i1 for (i1,i2,a,r,d,p) in minibatch]).to(device)
            rgb2_batch   = torch.cat([i2 for (i1,i2,a,r,d,p) in minibatch]).to(device)
            action_batch = torch.Tensor([a for (i1,i2,a,r,d,p) in minibatch])
            reward_batch = torch.Tensor([r for (i1,i2,a,r,d,p) in minibatch])
            done_batch   = torch.Tensor([d for (i1,i2,a,r,d,p) in minibatch])
            pcd_batch    = torch.Tensor([p for (i1,i2,a,r,d,p) in minibatch])

            Q1 = model1(rgb1_batch) 
            with torch.no_grad():
                Q2 = model2(rgb2_batch) 
            
            # Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0]) #N
            # X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            # loss = loss_fn(X, Y.detach())

            loss = criterion(pcd_batch)
            print(f"epoch : {epoch}, loss : {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            if episode_step % args.sync_freq == 0: 
                model2.load_state_dict(model1.state_dict())
                torch.save(model1.state_dict(), os.path.join(job_dir, f'{args.model_name}_{epoch}.pth'))

            if epoch % args.eval_freq == 0:
                acc = test_model(model1)
                print(f"validation accuracy : {acc}")
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model1.state_dict(), os.path.join(job_dir, 'best_accurracy.pth'))

        if reward != -1 or args.max_time > (time.time() - start_time):
            status = "stop"

client.enableApiControl(False)
losses = np.array(losses)
np.save("./results/losses.npy", losses)