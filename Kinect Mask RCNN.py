import pickle
import cv2
import numpy as np
import os
import sys
import coco
import utils
import model as modellib

writerC = cv2.VideoWriter('C:/Users/Paperspace/Desktop/Mask_RCNN-master/Mask_RCNN-master/Kinect_ColorR.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (1920, 1080))
writerD = cv2.VideoWriter('C:/Users/Paperspace/Desktop/Mask_RCNN-master/Mask_RCNN-master/Kinect_ColorD.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (512, 424))
capture = cv2.VideoCapture('C:/Users/Paperspace/Desktop/Mask_RCNN-master/Mask_RCNN-master/Kinect_Color.mp4')
filename = 'Kinect_Depth'
infile = open(filename, 'rb')
frameDD = pickle.load(infile)
Center = 0
frame_idx = 0
frame_idx1 = 0

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.print()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores,frame_idx):
    """
        take the image and results and apply the mask, box, and Label
    """

    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)
    depthxy = frameDD[frame_idx]
    frame_idx += 1
    if frame_idx == len( depthxy):
        frame_idx = 0
    depthxy = np.reshape(depthxy, (424, 512))
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        x_Center = ((x1 + x2) / 2) * 1.3
        y_Center = ((y1 + y2) / 2) * 1.3
        Pixel = depthxy[int(y_Center * .28)]
        Pixel_Depth = Pixel[int(x_Center / 3.5) - 100]
        textD = 'Depth {}mm'.format(Pixel_Depth)

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2
        )
        image = cv2.putText(
            image, textD, (x2 , y2), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2
        )

    return image

def display_instancesD(imageD, boxes,frame_idx):
    """
    take the image and results and apply the mask, box, and Label
    """

    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)
    frameD = frameDD[frame_idx]
    frame_idx += 1
    if frame_idx == len(frameD):
        frame_idx = 0
    frameD = frameD.astype(np.uint8)
    frameD = np.reshape(frameD, (424, 512))
    frameD = cv2.cvtColor(frameD, cv2.COLOR_GRAY2BGR)
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue


        y1, x1, y2, x2 = boxes[i]
        x_Center = ((x1 + x2) / 2) * 1.3
        y_Center = ((y1 + y2) / 2) * 1.3
        Center = int(x_Center / 3.5) - 100, int(y_Center * .28)
        imageD = cv2.circle(frameD, Center, 10, color, -1)

    return imageD

    # these 2 lines can be removed if you dont have a 1080p camera.
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:

    ret, frame = capture.read()
    results = model.detect([frame], verbose=0)
    r = results[0]
    frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],frame_idx)

    frameD = frameDD[frame_idx]
    frame_idx += 1
    if frame_idx == len(frameD):
        frame_idx = 0
    frameD = frameD.astype(np.uint8)
    frameD = np.reshape(frameD, (424, 512))
    frameD = cv2.cvtColor(frameD, cv2.COLOR_GRAY2BGR)
    frameR = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)


    frameD = display_instancesD(frameD, r['rois'],frame_idx)

    writerC.write(frame)
    writerD.write(frameD)
    cv2.imshow('frameD', frameD)
    cv2.imshow('frame', frameR)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()