import utils.setup_path as setup_path
import airsim

import torch

from tqdm import tqdm

from tools.vision import *
from tools.train_utils import *

def test(model, client, map_infos, args):
    with torch.no_grad():
        client = resetState(client)

        global_pcd = o3d.geometry.PointCloud()

        client.simPause(True)
        rgb1, depth1 = getImages(client) # 드론의 카메라를 통해 RGB 이미지를 얻음
        rgb1 = torch.from_numpy(rgb1).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(args.device)
        depth1 = torch.from_numpy(depth1).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(args.device)
        curr_pcd = get_transformed_lidar_pc(client) # 현재 포인트 클라우드를 전체 클라우드에 병합, 현재시점의 데이터를 전체 데이터에 누적
        global_pcd = global_pcd + curr_pcd # 현재 포인트 클라우드를 전체 클라우드에 복사
        client.simPause(False)

        status = "running"
        running_time = time.time()
        while(status == "running"): 
            qval = model(rgb1, depth1)
            qval = qval.cpu().data.numpy()

            x, y, z, radian, _ = calcValues(qval, args)
            move_start_time = time.time()   # 현재 시간을 측정, 드론이 움직이기 시작한 시간
            move_with_timeout(client, x, y, z, 3)  # 3초 타임아웃
            rotate_with_timeout(client, args["drone"].yaw_rate, radian, 3)  # 3초 타임아웃

            client.simPause(True)
            running_time += time.time() - move_start_time   # 드론이 움직인 시간을 측정
            rgb2, depth2 = getImages(client) # 드론의 카메라를 통해 RGB 이미지를 얻음
            rgb2 = torch.from_numpy(rgb2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(args.device)
            depth2 = torch.from_numpy(depth2).unsqueeze(dim=0).permute(0, 3, 1, 2).float().to(args.device)
            curr_pcd = get_transformed_lidar_pc(client)   # 현재 포인트 클라우드를 전체 클라우드에 병합, 현재시점의 데이터를 전체 데이터에 누적
            reward = calcReward(map_infos, global_pcd, curr_pcd, running_time, args)   # 보상을 계산, 맵의 포인트 클라우드, 전체 포인트 클라우드, 현재 포인트 클라우드, 드론이 움직인 시간, args를 인자로 전달
            client.simPause(False)

            global_pcd = global_pcd + curr_pcd    # 현재 포인트 클라우드를 전체 클라우드에 복사

            rgb1 = copy.deepcopy(rgb2)
            depth1 = copy.deepcopy(depth2)

            status = calcStatus(reward) # 보상을 기반으로 현재 에피소드 상태를 계산.

        SlamWell = True if status == "work well" else False
    return SlamWell

def getAccuracy(model, client, map_infos, logger, args):
    SlamWells = 0
    for _ in tqdm(range(args.eval_steps), desc = 'Validation'):
        SlamWells += test(model, client, map_infos, args)
    SlamWell_perc = float(SlamWells) / float(args.eval_steps)
    logger.info("Simulated steps: {0}, # of Validation accuracy: {1}".format(args.eval_steps, 100.0 * SlamWells))
    return 100.0 * SlamWell_perc 