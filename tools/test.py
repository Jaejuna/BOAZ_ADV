import utils.setup_path as setup_path
import airsim

import torch

from tqdm import tqdm

from tools.vision import *
from tools.train_utils import *

def test(model, client, map_pcd, args):
    pcd_global = o3d.geometry.PointCloud()
    pcd_global = merge_point_clouds(pcd_global, getPointCloud(client), client)

    rgb1 = getRGBImage(client)
    rgb1 = torch.from_numpy(rgb1).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(args.device)

    status = "running"
    start_time = time.time()
    while(status == "running"): 
        qval = model(rgb1)
        qval = qval.cpu().data.numpy()

        x, y, z, radian, _ = calcValues(qval, args)
        
        client.moveToPositionAsync(x, y, z, args["drone"].default_velocity, 
                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                   yaw_mode=airsim.YawMode(False, clacDuration(args["drone"].yaw_rate, radian))).join()

        curr_pcd = merge_point_clouds(pcd_global, getPointCloud(client), client)
        reward = calcReward(map_pcd, pcd_global, curr_pcd, client, args)
        pcd_global = copy.deepcopy(curr_pcd)

        rgb2 = getRGBImage(client)
        rgb2 = torch.from_numpy(rgb2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(args.device)
        rgb1 = rgb2

        if reward != -1:
            if reward > 0:
                status = "success"
            else:
                status = "fail"
            
        if args.max_time > (time.time() - start_time):
            break

    SlamWell = True if status == "success" else False
    return SlamWell

def getAccuracy(model, client, map_pcd, args):
    SlamWells = 0
    for _ in tqdm(range(args.eval_steps), desc = 'Validation'):
        SlamWell = test(model, client, map_pcd, args)
        if SlamWell:
            SlamWells += 1
    SlamWell_perc = float(SlamWells) / float(args.eval_steps)
    print("Simulated steps: {0}, # of Validation accuracy: {1}".format(args.eval_steps, 100.0 * SlamWells))
    return 100.0 * SlamWell_perc 