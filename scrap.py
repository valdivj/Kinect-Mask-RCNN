x_Center = int((((results(x1)) + (results['bottomright']['x'])) / 2))
y_Center = int((((results['topleft']['y']) + (results['bottomright']['y'])) / 2))
Center = (int(x_Center / 2), int(y_Center * .8))
Pixel = depthxy[int(y_Center * .8)]
Pixel_Depth = Pixel[int(x_Center / 2)]
# writerC.write(frame)
frameDD = frameD[frame_idx]
depthxy = frameD[frame_idx]
frame_idx += 1
if frame_idx == len(frame):
    frame_idx = 0
depthxy = np.reshape(depthxy, (424, 512))
frameDD = frameDD.astype(np.uint8)
frameDD = np.reshape(frameDD, (424, 512))
frameDD = cv2.cvtColor(frameDD, cv2.COLOR_GRAY2BGR)

cv2.imshow('frameD', frameDD)