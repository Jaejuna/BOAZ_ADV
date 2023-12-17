from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from tools.vision import *
from tools.train_utils import * 
from tools.dataset import ManualDataset

from config.default import args

# main 함수
if __name__ == '__main__':
    # python 디랙토리를 만들고 현재 날짜와 시간을 포함하는 디렉토리 경로를 생성
    date = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    job_dir = os.path.join("./run", date) + "_manual"
    os.makedirs(job_dir, exist_ok=True)

    logger = make_logger(job_dir)

    best_accuracy = 0
    device = args.device

    train_dataset = ManualDataset(args["manual"].train_txt_path, args["manual"].transform)
    test_dataset = ManualDataset(args["manual"].test_txt_path)
    train_loader = DataLoader(train_dataset, batch_size=args["manual"].batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["manual"].batch_size, shuffle=False)

    model, _ = create_models(args)
    criterion = torch.nn.MSELoss() # 손실함수 mse 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) # 최적화 알고리즘 아담사용

    for epoch in range(500):
        print_and_logging(logger, f"\n{epoch}epoch start...")
        for exp in train_loader:
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
            print_and_logging(logger, f"loss : {loss_move.item()}")

            optimizer.zero_grad()
            loss_move.backward()
            optimizer.step()

        with torch.no_grad():
            eq_count = 0
            for exp in test_loader:
                rgb1, rgb2, depth1, depth2, position, action, reward, done = exp

                rgb1 = rgb1.to(device).float()
                depth1 = depth1.to(device).float()
                position = position.to(device).float()
                action = action.to(device).float()

                Q = model(rgb1, depth1, position)

                softmax_batch = F.softmax(Q, dim=1).detach().cpu().numpy()
                Q_action = [np.random.choice(np.array([0, 1, 2]), p=softmax) for softmax in softmax_batch]
                Q_action = torch.FloatTensor(Q_action).to(device)

                eq_count += (action == Q_action).sum().item()

            accuracy = eq_count / len(test_dataset)
            print_and_logging(logger, f"test accuracy : {accuracy}")

        if best_accuracy < accuracy:
            print_and_logging(logger, "save new best model...")
            torch.save(model.state_dict(), os.path.join(job_dir, 'best_manual_model.pth')) 
            best_accuracy = accuracy

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join(job_dir, f'manual_trained_model_{epoch}epoch.pth')) 