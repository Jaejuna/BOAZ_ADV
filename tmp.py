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

def DepthConversion(PointDepth, f):
   H = PointDepth.shape[0]
   W = PointDepth.shape[1]
   i_c = np.float(H) / 2 - 1
   j_c = np.float(W) / 2 - 1
   columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
   DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
   PlaneDepth = PointDepth / (1 + (DistanceFromCenter / f)**2)**(0.5)
   return PlaneDepth.astype(np.uint16)

def getPointCloudByIntrinsic_(client, save=True):
   rgb_img, depth_img = getImages_(client)
   if save:
    cv2.imwrite("./results/rgb.png", rgb_img)
    cv2.imwrite("./results/depth.png", depth_img.astype('uint16'))

   fx, fy, cx, cy = get_intrinsic_()
   rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
   # depth_img = DepthConversion(depth_img, fx)

   # color = o3d.geometry.Image(rgb_img)
   # depth = o3d.geometry.Image(depth_img)

   # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
   #    color, depth, depth_scale=1000.0, depth_trunc=6.0, convert_rgb_to_intensity=False
   # )

   # intrinsics = o3d.camera.PinholeCameraIntrinsic(224, 224, fx, fy, cx, cy)
   # point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

   # Convert depth to 3D points
   u = np.linspace(0, 223, 224)
   v = np.linspace(0, 223, 224)
   u, v = np.meshgrid(u, v)

   x = (u - cx) * depth_img / fx
   y = (v - cy) * depth_img / fy
   z = depth_img.squeeze()  # Remove the third dimension

   # Convert coordinates using depth
   x = x * z
   y = y * z
   
   x = x.ravel()
   y = y.ravel()
   z = z.ravel()     

   # Create point cloud from the 3D points
   points = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))

   # Convert the point cloud to Open3D format
   point_cloud = o3d.geometry.PointCloud()
   point_cloud.points = o3d.utility.Vector3dVector(points)
   point_cloud.points = o3d.utility.Vector3dVector(rgb_img.reshape(-1, 3))

   return point_cloud

def get_intrinsic_():
   # 하드 코딩한 결과
   # 다른 방법 없음
   w = 224
   h = 224
   fov = 90

   fx = w / (2 * math.tan(fov / 2))
   fy = h / (2 * math.tan(fov / 2))
   cx = w / 2
   cy = h / 2

   return fx, fy, cx, cy

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

client.moveToPositionAsync(-10, 10, -10, 10).join()

# depth_to_point_cloud(client)
pcd = getPointCloudByIntrinsic_(client)
o3d.io.write_point_cloud("./results/pcd.ply", pcd)
