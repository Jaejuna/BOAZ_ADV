from torch.utils.data import Dataset
import torch

import cv2

import os

class ManualDataset(Dataset):
    def __init__(self, txt_path):
        self.root = "/".join(txt_path.split("/")[:-1]) + "/exp"
        self.ids = list()
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                self.ids.append(line.strip())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name_tag = self.ids[index]
            
        rgb1 = cv2.imread(os.path.join(self.root, f"rgb1_{name_tag}.jpg"))
        rgb2 = cv2.imread(os.path.join(self.root, f"rgb2_{name_tag}.jpg"))
        depth1 = cv2.imread(os.path.join(self.root, f"depth1_{name_tag}.jpg"))
        depth2 = cv2.imread(os.path.join(self.root, f"depth2_{name_tag}.jpg"))

        with open(self.root + f"values_{name_tag}.txt", "r") as f:
            line = f.readline()
        
        p1, p2, p3, action, reward, done = map(float, line.split())

        position = torch.FloatTensor([p1, p2, p3]).unsqueeze(dim=0)
        
        return (rgb1, rgb2, depth1, depth2, position, int(action), reward, bool(int(done)))

    def collate_fn(self, batch):
        rgb1 = torch.cat([i1 for (i1, i2, d1, d2, p, a, rw, d) in batch], dim=0)
        rgb2 = torch.cat([i2 for (i1, i2, d1, d2, p, a, rw, d) in batch], dim=0)
        depth1 = torch.cat([d1 for (i1, i2, d1, d2, p, a, rw, d) in batch], dim=0)
        depth2 = torch.cat([d2 for (i1, i2, d1, d2, p, a, rw, d) in batch], dim=0)
        position = torch.cat([p for (i1, i2, d1, d2, p, a, rw, d) in batch], dim=0)
        action = torch.Tensor([a for (i1, i2, d1, d2, p, a, rw, d) in batch])
        reward = torch.Tensor([rw for (i1, i2, d1, d2, p, a, rw, d) in batch])
        done = torch.Tensor([d for (i1, i2, d1, d2, p, a, rw, d) in batch])
  
        return (rgb1, rgb2, depth1, depth2, position, action, reward, done)