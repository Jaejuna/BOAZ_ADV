import utils.setup_path as setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

from tools.vision import *

def depth_to_point_cloud(client, fileName):
    # Get depth data from AirSim
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    depth_response = responses[0]
    
    # Convert depth data to 2D array
    depth_img_in_meters = airsim.list_to_2d_float_array(depth_response.image_data_float, depth_response.width, depth_response.height)
    depth_img_in_meters = depth_img_in_meters.reshape(depth_response.height, depth_response.width, 1)

    camera_info = client.simGetCameraInfo("0")  # Assuming "0" is the name of your camera
    proj_matrix = np.array(camera_info.proj_mat.matrix)

    # Use cv2 to convert depth image to 3D point cloud
    depth_16bit = np.clip(depth_img_in_meters * 1000, 0, 65535).astype('uint16')
    gray = cv2.cvtColor(depth_16bit, cv2.COLOR_BGR2GRAY)
    Image3D = cv2.reprojectImageTo3D(gray, proj_matrix)

    f = open(fileName, "w")
    for x in range(Image3D.shape[0]):
        for y in range(Image3D.shape[1]):
            pt = Image3D[x,y]
            if (math.isinf(pt[0]) or math.isnan(pt[0])):
                # skip it
                None
            else: 
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, "%d %d %d" % (0,255,0)))
    f.close()

    return point_cloud_to_o3d(Image3D)
    # # Connect to AirSim
    # responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    # depth_response = responses[0]
    
    # # Convert depth data to 2D array
    # depth_img_in_meters = airsim.list_to_2d_float_array(depth_response.image_data_float, depth_response.width, depth_response.height)
    # depth_img_in_meters = depth_img_in_meters.reshape(depth_response.height, depth_response.width, 1)
    
    # # Use cv2 to convert depth image to 3D point cloud
    # depth_16bit = np.clip(depth_img_in_meters * 1000, 0, 65535).astype(np.int16)

    # # Get camera intrinsic parameters from AirSim
    # camera_info = client.simGetCameraInfo("0")  # Assuming "0" is the name of your camera
    # proj_matrix = np.array(camera_info.proj_mat.matrix)
    # print(proj_matrix)
    # fx = proj_matrix[0]
    # fy = proj_matrix[5]
    # cx = proj_matrix[2]
    # cy = proj_matrix[6]

    # # Create meshgrid of image pixel coordinates
    # h, w = depth_16bit.shape
    # u = np.linspace(0, w-1, w)
    # v = np.linspace(0, h-1, h)
    # u, v = np.meshgrid(u, v)

    # # Compute 3D coordinates from depth
    # z = depth_16bit
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy

    # # Stack arrays into Nx3 point cloud
    # valid_depth = z > 0  # Consider only pixels with positive depth
    # point_cloud = np.stack((x[valid_depth], y[valid_depth], z[valid_depth]), axis=1)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))
    # o3d.io.write_point_cloud("results/pcd.ply", pcd)

    # return pcd



def getPointCloudForTest(client, fileName):
#    depthImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
#    png = cv2.imdecode(np.frombuffer(depthImage, np.uint8) , cv2.IMREAD_UNCHANGED)
#    gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
#    Image3D = cv2.reprojectImageTo3D(gray, get_transformation_matrix(client))

    # # Get depth data from AirSim
    # responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    # depth_response = responses[0]
    
    # # Convert depth data to 2D array
    # depth_img_in_meters = airsim.list_to_2d_float_array(depth_response.image_data_float, depth_response.width, depth_response.height)
    # depth_img_in_meters = depth_img_in_meters.reshape(depth_response.height, depth_response.width, 1)
    
    # # Use cv2 to convert depth image to 3D point cloud
    # depth_16bit = np.clip(depth_img_in_meters * 1000, 0, 65535).astype(np.int16)
    # # gray = cv2.cvtColor(depth_16bit, cv2.COLOR_BGR2GRAY)
    # Image3D = cv2.reprojectImageTo3D(depth_16bit, get_transformation_matrix(client))

    # f = open(fileName, "w")
    # for x in range(Image3D.shape[0]):
    #     for y in range(Image3D.shape[1]):
    #         pt = Image3D[x,y]
    #         if (math.isinf(pt[0]) or math.isnan(pt[0])):
    #             # skip it
    #             pass
    #         else: 
    #             f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, "%d %d %d" % (0,255,0)))
    # f.close()
    # Get depth data from AirSim
    # responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    # depth_response = responses[0]
    
    # # Convert depth data to 2D array
    # depth_img_in_meters = airsim.list_to_2d_float_array(depth_response.image_data_float, depth_response.width, depth_response.height)
    # depth_img_in_meters = depth_img_in_meters.reshape(depth_response.height, depth_response.width, 1)

    # camera_info = client.simGetCameraInfo("0")  # Assuming "0" is the name of your camera
    # proj_matrix = np.array(camera_info.proj_mat.matrix)

    # # Use cv2 to convert depth image to 3D point cloud
    # depth_16bit = np.clip(depth_img_in_meters * 1000, 0, 65535).astype('int16')
    # # gray = cv2.cvtColor(depth_16bit, cv2.COLOR_BGR2GRAY)
    # Image3D = cv2.reprojectImageTo3D(depth_16bit, proj_matrix)

    depthImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
    png = cv2.imdecode(np.frombuffer(depthImage, np.uint8) , cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
    camera_info = client.simGetCameraInfo("0")  # Assuming "0" is the name of your camera
    proj_matrix = np.array(camera_info.proj_mat.matrix)
    Image3D = cv2.reprojectImageTo3D(gray, proj_matrix)
    f = open(fileName, "w")
    for x in range(Image3D.shape[0]):
        for y in range(Image3D.shape[1]):
            pt = Image3D[x,y]
            if (math.isinf(pt[0]) or math.isnan(pt[0])):
                # skip it
                None
            else: 
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, "%d %d %d" % (0,255,0)))
    f.close()


def getImages(client):
    MIN_DEPTH_METERS = 0
    MAX_DEPTH_METERS = 100

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
    depth_img_in_meters = airsim.list_to_2d_float_array(depth_response.image_data_float, depth_response.width, depth_response.height)
    depth_img_in_meters = depth_img_in_meters.reshape(depth_response.height, depth_response.width, 1)

    # Lerp 0..100m to 0..255 gray values
    depth_8bit_lerped = np.interp(depth_img_in_meters, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255))
    cv2.imwrite("results/depth_visualization.png", depth_8bit_lerped.astype('uint8'))

    # Convert depth_img to millimeters to fill out 16bit unsigned int space (0..65535). Also clamp large values (e.g. SkyDome) to 65535
    depth_img_in_millimeters = depth_img_in_meters * 1000
    depth_16bit = np.clip(depth_img_in_millimeters, 0, 65535)
    cv2.imwrite("results/depth_16bit.png", depth_16bit.astype('uint16'))
   
    return rgb_img, depth_8bit_lerped, depth_16bit

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

client.moveToPositionAsync(-10, 10, -10, 10).join()

rgb, d1, d2 = getImages(client)
print(d1)
print()
print(d2)

cv2.imwrite("./results/rgb1.png", rgb)
# depth_to_point_cloud(client)
getPointCloudForTest(client, "./results/cloud1.asc")

# client.moveToPositionAsync(-10, 10, -10, 10).join()

# rgb = getImages(client)

# cv2.imwrite("./results/rgb2.png", rgb)

# getPointCloudForTest(client, "./results/cloud2.asc")

# import os
# import airsim

# client = airsim.MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True)

# client.armDisarm(True)
# client.takeoffAsync().join()

# binvox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps", "map.binvox")
# print("Create map voxel...")
# center = airsim.Vector3r(0, 0, 0)
# client.simCreateVoxelGrid(center, 1000, 1000, 100, 0.5, binvox_path)
# print("done!")