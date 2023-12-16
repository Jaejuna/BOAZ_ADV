import utils.setup_path as setup_path
import airsim

from collections import deque

import copy
import random
from datetime import datetime
import multiprocessing

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from tools.test import getAccuracy
from tools.models import MovePredictModel, RGBDepthFusionNet
from tools.vision import *
from tools.train_utils import * 
from tools.live_visualization import live_visualization
from tools.dataset import ManualDataset

from config.default import args

# main 함수
if __name__ == '__main__':
    # python 디랙토리를 만들고 현재 날짜와 시간을 포함하는 디렉토리 경로를 생성
    date = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    job_dir = os.path.join("./run", date) + "_manual"
    os.makedirs(job_dir, exist_ok=True)

    device = args.device # cpu or gpu

    dataset = ManualDataset(args["manual"].txt_path)
    data_loader = DataLoader(dataset, batch_size=args["manual"].batch_size, shuffle=True)

    model, _ = create_models(args)
    criterion = torch.nn.MSELoss() # 손실함수 mse 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) # 최적화 알고리즘 아담사용

    for exp in data_loader:
        rgb1, rgb2, depth1, depth2, position, action, reward, done = exp

        rgb1 = rgb1.to(device).float()
        depth1 = depth1.to(device).float()
        position = position.to(device).float()
        action = action.to(device).float()
        reward = reward.to(device).float()
        done = done.to(device).float()

        Q = model(rgb1, depth1, position)

        softmax_batch = F.softmax(Q, dim=1).detach().cpu().numpy()
        Q_action = [np.random.choice(np.array([0, 1, 2]), p=softmax) for softmax in softmax_batch]
        Q_action = torch.FloatTensor(Q_action).to(device)

        Q = Q.gather(dim=1, index=Q_action.long().unsqueeze(dim=1)).squeeze()
        Y = reward + args.gamma * ((1 - done) * action)

        loss_move = criterion(Q, Y.detach())
        print(f"loss : {loss_move.item()}")

        optimizer.zero_grad()
        loss_move.backward()
        optimizer.step()

        torch.save(model.state_dict(), os.path.join(job_dir, 'manual_trained_model.pth')) 