import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import binvox_rw

class VoxelMSELoss(nn.Module):
    def __init__(self, reference_point):
        super(VoxelMSELoss, self).__init__()
        self.reference_point = reference_point
        self.full_aligned = torch.roll(self.getMapVoxel(), shifts=self.reference_point, dims=(0, 1, 2))

    def getMapVoxel(self):
        binvox_path = os.path.join("results", "map.binvox")

        # 복셀 데이터 읽기
        with open(binvox_path, 'rb') as f:
            voxel_data = binvox_rw.read_as_3d_array(f)
        
        dims = voxel_data.dims
        tensor = torch.zeros(dims, dtype=torch.float32)
        for x in range(dims[0]):
            for y in range(dims[1]):
                for z in range(dims[2]):
                    tensor[x, y, z] = voxel_data.data[x, y, z]

        return tensor

    def forward(self, partial_voxels):
        partial_aligned = torch.roll(partial_voxels, shifts=self.reference_point, dims=(0, 1, 2))
        loss = F.mse_loss(self.full_aligned, partial_aligned)
        return loss
