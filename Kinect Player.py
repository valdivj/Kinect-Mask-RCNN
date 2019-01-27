from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2
import pickle
import time

# Same command function as streaming, its just now we pass in the file path, nice!

KinectC = cv2.VideoCapture('C:/Users/shirley/Desktop/Mask_RCNN-Kinect/Kinect_ColorR.mp4')
KinectD = cv2.VideoCapture('C:/Users/shirley/Desktop/Mask_RCNN-Kinect/Kinect_ColorD.mp4')
fps = 15

# Always a good idea to check if the video was acutally there
# If you get an error at thsi step, triple check your file path!!
if KinectC.isOpened() == False:
    print(
        "Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")

# While the video is opened
while KinectC.isOpened():

    # Read the video file.
    ret, frameC = KinectC.read()
    ret, frameD = KinectD.read()
    frameC = cv2.resize(frameC, (0, 0), fx=0.5, fy=0.5)
    # If we got frames, show them.
    if ret == True:


        # Display the frame at same frame rate of recording
        # Watch lecture video for full explanation
        time.sleep(1 / fps)
        cv2.imshow('frameC', frameC)
        cv2.imshow('frameD', frameD)
        # Press q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Or automatically break this whole loop if the video is over.
    else:
        break
KinectD.release()
KinectC.release()
# Closes all the frames
cv2.destroyAllWindows()