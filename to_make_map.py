import utils.setup_path as setup_path
import airsim

from math import pi
from datetime import datetime
import numpy as np
import cv2

from tools.vision import *
from tools.train_utils import *

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()


# global_pcd = o3d.geometry.PointCloud() # 새롭게 맵을 만들때 사용
global_pcd = o3d.io.read_point_cloud("./results/map_created_by_lidar.ply") # 기존에 만들던 맵을 계속해서 사용할 때 사용

while 1:
   input_data = int(input("input data : "))

   if input_data == 0:
      # client.moveToPositionAsync(*input_data, 5).join()
      forward = calcForwardDirection(client)
      client.moveByVelocityAsync(*list(map(lambda x:x*2, forward)), 2).join()
   elif input_data == 1 or input_data == 2:
      degree = calcDegree(client, input_data)
      client.rotateToYawAsync(degree).join()
   else:
      print("check your input")
      continue

# while 1:
#    input_data = input("input data : ")
#    if input_data == "break": break
#    elif input_data == "reset": 
#       client.reset()
#    try:
#       input_data = list(map(float, input_data.split(" ")))
#    except:
#       print("something wrong...")
#       continue

#    if len(input_data) == 3:
#       # client.moveToPositionAsync(*input_data, 5).join()
#       client.moveByVelocityAsync(*input_data, 1).join()
#    elif len(input_data) == 1:
#       client.rotateToYawAsync(*input_data).join()
#    else:
#       print("check your input")
#       continue
#    state = client.getMultirotorState()
#    orientation = state.kinematics_estimated.orientation
#    _, _, yaw = airsim.to_eularian_angles(orientation)

#    client.simPause(True)
#    pcd, state = get_transformed_lidar_pc(client)
#    global_pcd = global_pcd + pcd
#    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)
#    pose = client.simGetVehiclePose()
#    print(pose.position.x_val, pose.position.y_val, pose.position.z_val)
#    client.simPause(False)
#    o3d.io.write_point_cloud("./results/map_created_by_lidar.ply", global_pcd)
