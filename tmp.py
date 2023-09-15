import utils.setup_path as setup_path
import airsim

import numpy as np
import cv2

from tools.vision import *

def getImages_(client):
   # Request DepthPerspective image as uncompressed float
   responses = client.simGetImages(
      [
         airsim.ImageRequest("0", airsim.ImageType.Scene , False, False),
         airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
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

def savepointcloud(image, filename):
   f = open(filename, "w+")
   for x in range(image.shape[0]):
      for y in range(image.shape[1]):
         pt = image[x, y]
         if math.isinf(pt[0]) or math.isnan(pt[0]) or pt[0] > 10000 or pt[1] > 10000 or pt[2] > 10000:
               None
         else:
               f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2] - 1, "0 255 0"))
   f.close()

def DepthConversion(PointDepth, f):
   H = PointDepth.shape[0]
   W = PointDepth.shape[1]
   i_c = np.float(H) / 2 - 1
   j_c = np.float(W) / 2 - 1
   columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
   DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
   PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
   return PlaneDepth

def generatepointcloud(depth):
   Fx, Fy, Cx, Cy = get_intrinsic_()
   rows, cols = depth.shape
   c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
   valid = (depth > 0) & (depth < 255)
   z = 1000 * np.where(valid, depth / 256.0, np.nan)
   x = np.where(valid, z * (c - Cx) / Fx, 0)
   y = np.where(valid, z * (r - Cy) / Fy, 0)
   return np.dstack((x, y, z))

def getPointCloudByIntrinsic_(client, save=True):
   responses = client.simGetImages(
      [
         airsim.ImageRequest("0", airsim.ImageType.Scene , False, False),
         airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),
      ]
   )
   rgb_response, depth_response = responses[0], responses[1]

   rgb = np.fromstring(rgb_response.image_data_uint8, dtype=np.uint8) 
   rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)
   rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

   depth = np.array(depth_response.image_data_float, dtype=np.float)
   depth[depth > 255] = 255
   depth = np.reshape(depth, (depth_response.height, depth_response.width))
   points1 = generatepointcloud(depth)
   depth = DepthConversion(depth, rgb_response.height / (2 * math.tan(90 / 2)))
   points2 = generatepointcloud(depth)
   points = (points1 + points2) / 2
   
   mask = ~np.isnan(points).any(axis=2) & ~np.isinf(points).any(axis=2)
   points = points[mask]
   rgb = rgb[mask]

   point_cloud = o3d.geometry.PointCloud()
   point_cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
   point_cloud.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
   point_cloud = point_cloud.voxel_down_sample(voxel_size=0.04) 
   # savepointcloud(pcl, r'C:\Users\USER\Desktop\boaz\adv\RL4AirSim\results\pcl.asc')
   return point_cloud

def get_intrinsic_(): 
   w = 224
   h = 224
   fov = 90

   fx = w / (2 * math.tan(fov / 2))
   fy = h / (2 * math.tan(fov / 2))
   cx = w / 2
   cy = h / 2

   return (fx, fy, cx, cy)

def mergePointClouds(cloud1, cloud2, client):
   source = copy.deepcopy(cloud1)
   target = copy.deepcopy(cloud2)

   voxel_size=0.04
   distance_threshold = voxel_size 
   radius_normal = voxel_size * 2
   radius_feature = voxel_size * 3
   max_nn = 30

   # coarse-to-fine manner의 Iterative Closest Point(ICP) 알고리즘을 사용하여 두 포인트 클라우드를 정합
   source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
   source_feature = o3d.pipelines.registration.compute_fpfh_feature(
         source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))
   
   target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))
   target_feature = o3d.pipelines.registration.compute_fpfh_feature(
         target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn))

   result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
         source, target, source_feature, target_feature, True, distance_threshold,
         o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
         [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.3),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
         o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.99))

   refinement_result =o3d.pipelines.registration.registration_icp(
               source, target, distance_threshold, result.transformation,
               o3d.pipelines.registration.TransformationEstimationPointToPlane())

   # 변환 행렬을 사용하여 source 포인트 클라우드를 변환
   target = copy.deepcopy(target).transform(np.linalg.inv(refinement_result.transformation))

   # 변환된 포인트 클라우드와 target 포인트 클라우드를 결합
   merged_pcd = copy.deepcopy(source + target)
   merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.04) 
   return merged_pcd

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

# client.moveToPositionAsync(-10, 10, -10, 10).join()

# depth_to_point_cloud(client)
client.moveToPositionAsync(-10, 10, -10, 10).join()
pcd1 = getPointCloudByIntrinsic_(client)
o3d.io.write_point_cloud(f"./results/pcd1.ply", pcd1)
client.moveToPositionAsync(-10, -10, -10, 10).join()
pcd2 = getPointCloudByIntrinsic_(client)
o3d.io.write_point_cloud(f"./results/pcd2.ply", pcd2)
print("merging")
pcd3 = mergePointClouds(pcd1, pcd2, client)
o3d.io.write_point_cloud(f"./results/pcd3.ply", pcd3)
print("done")