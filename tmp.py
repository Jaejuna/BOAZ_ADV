import utils.setup_path as setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

from vision import *

def getPointCloudForTest(client, fileName):
   depthImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
   png = cv2.imdecode(np.frombuffer(depthImage, np.uint8) , cv2.IMREAD_UNCHANGED)
   gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
   Image3D = cv2.reprojectImageTo3D(gray, get_transformation_matrix(client))

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
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False),
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
   
    return rgb_img

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

client.moveToPositionAsync(-10, 10, -10, 10).join()

rgb = getImages(client)

cv2.imwrite("./results/rgb1.png", rgb)

getPointCloudForTest(client, "./results/cloud1.asc")

client.moveToPositionAsync(-10, 10, -10, 10).join()

rgb = getImages(client)

cv2.imwrite("./results/rgb2.png", rgb)

getPointCloudForTest(client, "./results/cloud2.asc")

