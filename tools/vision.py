# use open cv to create point cloud from depth image.
import utils.setup_path as setup_path 
import airsim

import os
import time
import math
import copy

import torch
import cv2
import numpy as np
import open3d as o3d

from config.default import args

def savePointCloud(pcd, fileName):
   o3d.io.write_point_cloud(fileName, pcd)

def getRGBImage(client):
   # Request DepthPerspective image as uncompressed float
   responses = client.simGetImages(
      [
         airsim.ImageRequest("0", airsim.ImageType.Scene , False, False)
      ]
   )
   rgb_response = responses[0]

   # get numpy array
   img1d = np.fromstring(rgb_response.image_data_uint8, dtype=np.uint8) 

   # reshape array to 4 channel image array H X W X 3
   rgb_img = img1d.reshape(rgb_response.height, rgb_response.width, 3)

   return rgb_img

def getImages(client):
    # Request images from the AirSim client
    responses = client.simGetImages(
        [
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
        ]
    )
    rgb_response, depth_response = responses[0], responses[1]

    # Process and normalize the RGB image
    img1d = np.fromstring(rgb_response.image_data_uint8, dtype=np.uint8)
    rgb_img = img1d.reshape(rgb_response.height, rgb_response.width, 3)
    rgb_img = rgb_img.astype(np.float32) / 255.0  # Normalize to 0-1

    # Process the Depth image without converting to 16-bit and normalize it
    depth_img = airsim.list_to_2d_float_array(depth_response.image_data_float, depth_response.width, depth_response.height)
    depth_img = depth_img.reshape(depth_response.height, depth_response.width, 1)
    depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())  # Normalize to 0-1

    # Duplicate the depth channel to create a 3-channel image
    depth_img = np.repeat(depth_img, 3, axis=2)

    return rgb_img, depth_img

def getPointCloudByIntrinsic(client):
   rgb_img, depth_img = getImages(client)

   color = o3d.geometry.Image(rgb_img)
   depth = o3d.geometry.Image(depth_img)

   rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
      color, depth, depth_scale=1000.0, depth_trunc=40.0, convert_rgb_to_intensity=False
   )

   intrinsics = o3d.camera.PinholeCameraIntrinsic(1280, 720, *get_intrinsic())
   point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
   return point_cloud

def get_intrinsic():
   # 하드 코딩한 결과
   # 다른 방법 없음
   w = 224
   h = 224 
   fov = 90

   fx = w / (2 * math.tan(fov / 2))
   fy = h / (2 * math.tan(fov / 2))
   cx = w / 2
   cy = h / 2

   return (fx, fy, cx, cy)

def getPointCloud(client):
   rgb_img, depth_img = getImages(client)
   point_cloud = cv2.reprojectImageTo3D(depth_img, get_proj_matrix(client))
   return point_cloud_to_o3d(point_cloud, rgb_img)

def point_cloud_to_o3d(point_cloud, rgb_img):
   flattened_rgb = rgb_img.reshape(-1, 3)
   pcd = o3d.geometry.PointCloud()
   pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))
   pcd.colors = o3d.utility.Vector3dVector(flattened_rgb / 255.0)
   return pcd

def get_proj_matrix(client):
   # 드론의 현재 위치 및 자세 정보를 얻음
   camera_info = client.simGetCameraInfo("0")
   proj_matrix = np.array(camera_info.proj_mat.matrix)
   return proj_matrix

def mergePointClouds(cloud1, cloud2, client):
   source = copy.deepcopy(cloud1)
   target = copy.deepcopy(cloud2)

   # coarse-to-fine manner의 Iterative Closest Point(ICP) 알고리즘을 사용하여 두 포인트 클라우드를 정합
   threshold = 0.02
   T = get_proj_matrix(client)
   reg_p2p = o3d.pipelines.registration.registration_icp(
                     source, target, threshold, T, 
                     o3d.pipelines.registration.TransformationEstimationPointToPoint())

   # 변환 행렬을 사용하여 source 포인트 클라우드를 변환
   source.transform(reg_p2p.transformation)

   # 변환된 포인트 클라우드와 target 포인트 클라우드를 결합
   merged_pcd = source + target
   return merged_pcd

# def pcd_to_voxel_tensor(pcd, infos=None):
#    # 포인트 클라우드에서 복셀 그리드 생성
#    voxel_size = args.voxel_size
#    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
   
#    if infos is not None:
#       min_bound, max_bound, dims = infos

#    else:
#       # 복셀 그리드의 바운딩 박스 얻기
#       min_bound = voxel_grid.get_min_bound() // voxel_size
#       max_bound = voxel_grid.get_max_bound() // voxel_size
#       dims = np.ceil(max_bound - min_bound).astype(np.int64)

#    # 복셀의 중심 포인트 얻기
#    centers = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.float32)
#    indices = np.round((centers - min_bound) / voxel_size).astype(np.int64) 

#    # 빈 텐서(모든 값이 0) 생성
#    tensor = torch.zeros(*dims, dtype=torch.float32)

#    # 텐서에 복셀 값 설정
#    indices[:, 0] = np.clip(indices[:, 0], 0, dims[0]-1)
#    indices[:, 1] = np.clip(indices[:, 1], 0, dims[1]-1)
#    indices[:, 2] = np.clip(indices[:, 2], 0, dims[2]-1)
#    tensor[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
#    return tensor, [min_bound, max_bound, dims]

def pcd_to_voxel_tensor(pcd, infos=None):
   # 포인트 클라우드에서 복셀 그리드 생성
   voxel_size = args.voxel_size
   voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
   
   if infos is not None:
      min_bound, max_bound, dims = infos
   else:
      # 복셀 그리드의 바운딩 박스 얻기
      min_bound = voxel_grid.get_min_bound() // voxel_size
      max_bound = voxel_grid.get_max_bound() // voxel_size
      dims = np.ceil((max_bound - min_bound) / voxel_size).astype(np.int64)

   # 복셀의 중심 포인트 얻기
   centers = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()], dtype=np.float32)
   indices = np.round((centers - min_bound) / voxel_size).astype(np.int64) 

   # 빈 텐서(모든 값이 0) 생성
   tensor = torch.zeros(*dims, dtype=torch.float32)

   # 텐서에 복셀 값 설정
   for i, j, k in indices:
      if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
         tensor[i, j, k] += 1.0

   return tensor, [min_bound, max_bound, dims]

def calc_density(points):
   grid_size = 0.5  # 예시로 0.5 단위 크기의 그리드 사용
   min_bound = points.min(axis=0)
   
   grid_counts = {}
   for point in points:
      grid_index = np.floor((point - min_bound) / grid_size)
      grid_index = tuple(grid_index.astype(int))
      if grid_index not in grid_counts:
         grid_counts[grid_index] = 1
      else:
         grid_counts[grid_index] += 1

   densities = []
   for count in grid_counts.values():
      density = count / (grid_size ** 3)  # 밀도 = 포인트 수 / 그리드 부피
      densities.append(density)
   
   density_mean = np.mean(densities, axis=0)
   density_std = np.std(densities, axis=0)
   return density_mean, density_std

def getMapVoxel(map_path):
   pcd = o3d.io.read_point_cloud(map_path)
   pcd, map_infos = pcd_to_voxel_tensor(pcd)
   return pcd, map_infos

def getMapPointCloud(map_path):
   pcd = o3d.io.read_point_cloud(map_path)
   return pcd

def extract_infos_from_pcd(pcd):
   points = pcd.points
   points_array = np.array(points)

   len_pcd = len(points)
   dist_mean = np.mean(points_array, axis=0)
   dist_std = np.std(points_array, axis=0)
   density_mean, density_std = calc_density(points_array)

   return [len_pcd, dist_mean, dist_std, density_mean, density_std]

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
