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
   return pcd.transform(transformation_matrix)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

global_pcd = o3d.geometry.PointCloud()

client.moveToPositionAsync(2, 2, -2, 5).join()
client.simPause(True)
pcd1 = get_transformed_lidar_pc(client)
global_pcd = global_pcd + pcd1
client.simPause(False)

client.moveToPositionAsync(4, 6, -2, 5).join()
client.simPause(True)
pcd2 = get_transformed_lidar_pc(client)
global_pcd = global_pcd + pcd2
client.simPause(False)

client.moveToPositionAsync(4, 20, -2, 5).join()
client.simPause(True)
pcd3 = get_transformed_lidar_pc(client)
global_pcd = global_pcd + pcd3
client.simPause(False)

client.moveToPositionAsync(4, 29, -2, 5).join()
client.simPause(True)
pcd4 = get_transformed_lidar_pc(client)
global_pcd = global_pcd + pcd4
client.simPause(False)

client.rotateToYawAsync(90).join()
client.simPause(True)
pcd5 = get_transformed_lidar_pc(client)
global_pcd = global_pcd + pcd5
client.simPause(False)

client.moveToPositionAsync(15, 29, -2, 5).join()
client.simPause(True)
pcd6 = get_transformed_lidar_pc(client)
global_pcd = global_pcd + pcd6
client.simPause(False)

o3d.io.write_point_cloud(f"./results/pcd_lidar1.ply", pcd1)
o3d.io.write_point_cloud(f"./results/pcd_lidar2.ply", pcd2)
o3d.io.write_point_cloud(f"./results/pcd_lidar3.ply", pcd3)
o3d.io.write_point_cloud(f"./results/pcd_lidar4.ply", pcd4)
o3d.io.write_point_cloud(f"./results/pcd_lidar5.ply", pcd3)
o3d.io.write_point_cloud(f"./results/pcd_lidar6.ply", pcd4)
o3d.io.write_point_cloud(f"./results/pcd_lidar7.ply", global_pcd)