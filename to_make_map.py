import utils.setup_path as setup_path
import airsim

from datetime import datetime
import numpy as np
import cv2

from tools.vision import *

def get_transformed_lidar_pc(client):
   # raw lidar data to open3d PointCloud
   lidar_data = client.getLidarData()
   lidar_points = np.array([lidar_data.point_cloud[i:i+3] for i in range(0, len(lidar_data.point_cloud), 3)])
   pcd = o3d.geometry.PointCloud()
   pcd.points = o3d.utility.Vector3dVector(lidar_points)

   # calc rotation matrix
   orientation = lidar_data.pose.orientation
   q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
   rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                               [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                               [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)]))
   
   # calc translation vector
   position = lidar_data.pose.position
   x_val, y_val, z_val = position.x_val, position.y_val, position.z_val
   translation_vector = np.array([x_val, y_val, z_val])

   # apply transform
   transformation_matrix = np.eye(4)
   transformation_matrix[:3, :3] = rotation_matrix
   transformation_matrix[:3, 3] = translation_vector
   return pcd.transform(transformation_matrix), list(map(str, [q0, q1, q2, q3, x_val, y_val, z_val]))

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

global_pcd = o3d.geometry.PointCloud() # 새롭게 맵을 만들때 사용
# global_pcd = o3d.io.read_point_cloud("./results/map_created_by_lidar.ply") # 기존에 만들던 맵을 계속해서 사용할 때 사용

while 1:
   input_data = input("input data : ")
   if input_data == "break": break
   elif input_data == "reset": 
      client.reset()
   try:
      input_data = list(map(int, input_data.split(" ")))
   except:
      print("something wrong...")
      continue

   if len(input_data) == 3:
      print(f"move to {input_data}")
      client.moveToPositionAsync(*input_data, 5).join()
   elif len(input_data) == 1:
      print(f"rotate to {input_data}")
      client.rotateToYawAsync(*input_data).join()
   else:
      print("check your input")
      continue

   client.simPause(True)
   pcd, state = get_transformed_lidar_pc(client)
   global_pcd = global_pcd + pcd
   global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)
   client.simPause(False)
   o3d.io.write_point_cloud("./results/map_created_by_lidar.ply", global_pcd)
