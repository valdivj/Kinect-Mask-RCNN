# Kinect Mask RCNN

First of all I would recomend Looking at this repository:https://github.com/matterport/Mask_RCNN.
This will help you setup the MASK RCNN model on your computer.
My computer is not powerful enough to run the MASK RCNN model so I needed a way to
record my Kinect depth & Video stream and run it on my PaperSpace machine.

1.kinect record.py :This program records the kinect video stream as a mp4 file named "Kinect_Color.mp4".
And the depth stream as a Pickle file named "Kinect_Depth". Make sure you have the paths set to where you want the  video and depth stream stored.

2. Kinect Mask RCNN.py : Make sure you have the paths set to where the video and depth stream were stored.
When you run the program it  takes the recorded kinect depth and video stream and runs it through the Mask-RCNN model.
It finds the objects and puts bounding boxes,labels and masks on images. It also takes the center of the bounding boxes and uses that info to find the location of the object on the depth stream and extracts the depth reading and displays it on the bottom
right area of the bounding boxes.

The model runs at about 5 frames per second when its processing a recorded or live stream from the kinect.

As the model is processing the Kinect depth and video stream it is also saving them as "Kinect_ColorD.mp4" for the depth stream and "Kinect_ColorR.mp4" for the color stream.

3. Kinect Player.py: this plays back "Kinect_ColorD.mp4" and "Kinect_ColorR.mp4" at the same time at a regular frame rate so you can view the results of the model.Make sure you have the paths set to where the video and depth stream .mp4 files were stored

![Image description](https://github.com/valdivj/Kinect-Mask-RCNN/blob/master/Kinect_Mask.JPG)



