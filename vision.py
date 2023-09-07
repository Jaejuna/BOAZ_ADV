# use open cv to create point cloud from depth image.
import setup_path 
import airsim

import os
import time
import math

import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

# Constants for visualization
MIN_DEPTH_METERS = 0
MAX_DEPTH_METERS = 100
   
projectionMatrix = np.array([[-0.501202762, 0.000000000, 0.000000000, 0.000000000],
                              [0.000000000, -0.501202762, 0.000000000, 0.000000000],
                              [0.000000000, 0.000000000, 10.00000000, 100.00000000],
                              [0.000000000, 0.000000000, -10.0000000, 0.000000000]])

def savePointCloud(image, fileName, color=(0,255,0)):
   f = open(fileName, "w")
   for x in range(image.shape[0]):
     for y in range(image.shape[1]):
        pt = image[x,y]
        if (math.isinf(pt[0]) or math.isnan(pt[0])):
          # skip it
          None
        else: 
          f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, "%d %d %d" % color))
   f.close()

def getImages(client):
   # Request DepthPerspective image as uncompressed float
   responses = client.simGetImages(
      [
         airsim.ImageRequest("0", airsim.ImageType.Scene , False, False),
         airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
      ]
   )
   rgb_response, depth_response = responses[0], responses[1]

   # get numpy array
   img1d = np.fromstring(rgb_response.image_data_uint8, dtype=np.uint8) 

   # reshape array to 4 channel image array H X W X 3
   rgb_img = img1d.reshape(rgb_response.height, rgb_response.width, 3)

   # Reshape to a 2d array with correct width and height
   depth_img = airsim.list_to_2d_float_array(depth_response.image_data_float, depth_response.width, depth_response.height)
   depth_img = depth_img.reshape(depth_response.height, depth_response.width, 1)

   # Convert depth_img to millimeters to fill out 16bit unsigned int space (0..65535). Also clamp large values (e.g. SkyDome) to 65535
   depth_img = depth_img * 1000
   depth_img = np.clip(depth_img, 0, 65535).astype(np.uint16)

   return rgb_img, depth_img

def getPointCloud(client):
   depthImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
   png = cv2.imdecode(np.frombuffer(depthImage, np.uint8) , cv2.IMREAD_UNCHANGED)
   gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
   Image3D = cv2.reprojectImageTo3D(gray, projectionMatrix)
   return Image3D 

def pointCloudReconstruction():
   demo_icp_pcds = o3d.data.OfficePointClouds()

   voxel_size=0.04
   distance_threshold = voxel_size 
   radius_normal = voxel_size * 2
   radius_feature = voxel_size * 3
   max_nn = 30

   pcd_global = o3d.geometry.PointCloud()
   prev_pcd = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
   prev_pcd = prev_pcd.voxel_down_sample(voxel_size=voxel_size) 
   prev_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
   prev_feature = o3d.pipelines.registration.compute_fpfh_feature(
         prev_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))
   pcd_global = pcd_global + prev_pcd
   for i in range(1, 30):
      curr_pcd = o3d.io.read_point_cloud(demo_icp_pcds.paths[i])
      curr_pcd = curr_pcd.voxel_down_sample(voxel_size=voxel_size) 
      curr_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))

      curr_feature = o3d.pipelines.registration.compute_fpfh_feature(
               curr_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))

      result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
               prev_pcd, curr_pcd, prev_feature, curr_feature, True, distance_threshold,
               o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
               [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.3),
               o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
               o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.99))

      refinement_result =o3d.pipelines.registration.registration_icp(
                  prev_pcd, curr_pcd, distance_threshold, result.transformation,
                  o3d.pipelines.registration.TransformationEstimationPointToPlane())

      T_ref = refinement_result.transformation
      curr_pcd = copy.deepcopy(curr_pcd).transform(np.linalg.inv(T_ref))
      curr_pcd = o3d.geometry.PointCloud(curr_pcd)
      pcd_global = pcd_global + curr_pcd

      prev_pcd = copy.deepcopy(curr_pcd)
      prev_feature = copy.deepcopy(curr_feature)
   o3d.io.write_point_cloud("office.ply", pcd_global)