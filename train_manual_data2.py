from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from tools.vision import *
from tools.train_utils import * 
from tools.dataset import ManualDataset

from config.default import args

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

def epsilon_by_epoch(epoch):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * epoch / epsilon_decay)

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
        epsilon = epsilon_by_epoch(epoch)
        print_and_logging(logger, f"\n{epoch}epoch start... \nEpsilon: {epsilon}")

        for exp in train_loader:
            rgb1, rgb2, depth1, depth2, position, action, reward, done = exp

            # 현재 상태와 다음 상태를 장치에 전송
            rgb1 = rgb1.to(device).float()
            rgb2 = rgb2.to(device).float()
            depth1 = depth1.to(device).float()
            depth2 = depth2.to(device).float()
            position = position.to(device).float()
            action = action.to(device).long()
            reward = reward.to(device).float()
            done = done.to(device).float()

            # 현재 상태에 대한 Q값
            Q_current = model(rgb1, depth1, position)

            # 다음 상태에 대한 Q값
            Q_next = model(rgb2, depth2, position).detach()
            max_Q_next = Q_next.max(1)[0]

            # 목표 Q값 계산
            Y = reward + args.gamma * max_Q_next * (1 - done)

            # 행동 선택: 엡실론-탐욕적 전략
            Q_action = Q_current.max(1)[1]  # 최대 Q값을 가진 행동
            # if random.random() > epsilon:
            #     Q_action = Q_current.max(1)[1]  # 최대 Q값을 가진 행동
            # else:
            #     Q_action = torch.randint(0, 3, (rgb1.size(0),)).to(device)  # 무작위 행동

            # 선택된 행동에 대한 Q값
            Q_selected = Q_current.gather(1, Q_action.unsqueeze(1)).squeeze(1)

            # 손실 계산 및 역전파
            loss = criterion(Q_selected, Y)
            print_and_logging(logger, f"loss : {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
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