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