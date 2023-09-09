import utils.setup_path as setup_path
import airsim

import torch

from tools.vision import *
from tools.train_utils import *

def test(model, client, map_pcd, args):
    pcd_global = o3d.geometry.PointCloud()
    pcd_global = merge_point_clouds(pcd_global, getPointCloud(client), client)

    rgb1 = getRGBImage(client)
    rgb1 = torch.from_numpy(rgb1).to(args.device)

    status = "running"
    start_time = time.time()
    while(status == "running"): 
        qval = model(rgb1)
        qval = qval.data.numpy()

        x, y, z, radian, _ = calcValues(qval, args.epsilon)
        
        client.moveToPositionAsync(x, y, z, args["drone"].defualt_velocity, 
                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                   yaw_mode=airsim.YawMode(False, clacDuration(args["drone"].yaw_rate, radian))).join()

        curr_pcd = merge_point_clouds(pcd_global, getPointCloud(client), client)
        reward = calcReward(map_pcd, pcd_global, curr_pcd, client)
        pcd_global = copy.deepcopy(curr_pcd)

        rgb2 = getRGBImage(client)
        rgb2 = torch.from_numpy(rgb2).to(args.device)
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

def getAccuracy(model, client, map_pcd, args, max_games=10000):
    SlamWells = 0
    for _ in range(max_games):
        SlamWell = test(model, client, map_pcd, args)
        if SlamWell:
            SlamWells += 1
    SlamWell_perc = float(SlamWells) / float(max_games)
    print("Simulated times: {0}, # of SlamWell percentage: {1}".format(max_games, 100.0 * SlamWells))
    return 100.0 * SlamWell_perc 