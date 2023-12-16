from torch.utils.data import Dataset
import torch

import numpy as np

import os

class ManualDataset(Dataset):
    def __init__(self, txt_path):
        self.root = "data"
        self.ids = list()
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                self.ids.append(line.strip().split("/"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        date, name_tag = self.ids[index]
        
        rgb1 = np.load(os.path.join(self.root, date, f"rgb1_{name_tag}.npy"))[0]
        rgb2 = np.load(os.path.join(self.root, date, f"rgb2_{name_tag}.npy"))[0]
        depth1 = np.load(os.path.join(self.root, date, f"depth1_{name_tag}.npy"))[0]
        depth2 = np.load(os.path.join(self.root, date, f"depth2_{name_tag}.npy"))[0]

        with open(os.path.join(self.root, date, f"values_{name_tag}.txt"), "r") as f:
            line = f.readline()
        
        p1, p2, p3, action, reward, done = map(float, line.split())

        position = torch.FloatTensor([p1, p2, p3])
        
        return (rgb1, rgb2, depth1, depth2, position, int(action), reward, int(done))

    def collate_fn(self, batch):

        rgb1 = list()
        rgb2 = list()
        depth1 = list()
        depth2 = list()
        position = list()
        action = list()
        reward = list()
        done = list()

        for b in batch:
            rgb1.append(b[0])
            rgb2.append(b[1])
            depth1.append(b[2])
            depth2.append(b[3])
            position.append(b[4])
            action.append(b[5])
            reward.append(b[6])
            done.append(b[7])

        rgb1 = torch.stack(rgb1, dim=0)
        rgb2 = torch.stack(rgb2, dim=0)
        depth1 = torch.stack(depth1, dim=0)
        depth2 = torch.stack(depth2, dim=0)
        position = torch.stack(position, dim=0)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(action)
        done = torch.FloatTensor(done)
  
        return (rgb1, rgb2, depth1, depth2, position, action, reward, done)