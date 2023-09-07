from collections import deque

import copy
import random
from IPython.display import clear_output

import torch
import numpy as np
from math import pi

import setup_path
import airsim

from model import MovePredictModel
from vision import *
from config import args
from train_utils import *


device = args.device
yaw_rate = args["drone"].yaw_rate

model1 = MovePredictModel(num_classes=args.num_classes, backbone="resnet101")
model2 = copy.deepcopy(model1) 
model2.load_state_dict(model1.state_dict()) 
model1.to(device)
model2.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=args.learning_rate)

replay = deque(maxlen=args.mem_size)
losses = []
episode_step=0

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

for epoch in range(args.epochs):
    rgb, depth = getImages(client)

    client.hoverAsync().join()

    rgb, depth = getImages(client)
    pointCloud = getPointCloud(client)

    # game = Gridworld(size=4, mode='random')
    # state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
    # state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0
    while(status == 1): 
        episode_step += 1
        mov += 1
        qval = model1(rgb)
        qval_ = qval.data.numpy()
        if (random.random() < args.epsilon):
            action = np.random.randint(0, args.n_classes)
        else:
            action = np.argmax(qval_)
        
        client.rotateByYawRateAsync(yaw_rate, clacDuration(yaw_rate, )).join()
        client.moveToPositionAsync(-10, 10, -10, args.defualt_velocity).join()

        # game.makeMove(action)
        # state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        # state2 = torch.from_numpy(state2_).float()
        rgb, depth = getImages(client)
        reward = game.reward()
        done = True if reward > 0 else False
        exp =  (state1, action_, reward, state2, done)
        replay.append(exp) 
        state1 = state2
        
        if len(replay) > args.batch_size:
            minibatch = random.sample(replay, args.batch_size)
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
            Q1 = model1(state1_batch) 
            with torch.no_grad():
                Q2 = model2(state2_batch) 
            
            Y = reward_batch + args.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(epoch, loss.item())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            if episode_step % args.sync_freq == 0: 
                model2.load_state_dict(model1.state_dict())
        if reward != -1 or mov > max_moves:
            status = 0
            mov = 0
    
    client.reset()
        
client.enableApiControl(False)
losses = np.array(losses)
np.save("./results/losses.npy", losses)