import cv2
import numpy as np
from realsense_camera import *
import time

# Load Realsense camera
rs = RealsenseCamera()

# Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow(
    "dnn/frozen_inference_graph_coco.pb", "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Generate random colors(80 colors equals to detectable classes of the model, 3 is the number of channels)
colors = np.random.randint(0, 255, (80, 3))


while True:
    # Load image
    # img = cv2.imread('road.jpg')
    # height, width, _ = img.shape

    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    height, width, _ = bgr_frame.shape

    # Create a background image
    background = np.zeros((height, width, 3), np.uint8)
    background[:] = (100, 100, 0)
    # Detect objects
    blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
    net.setInput(blob)
    start = time.time()
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()
    detection_count = boxes.shape[2]
    print("[Info] Mask R-CNN took {:.6f} seconds".format(end - start))

    for i in range(detection_count):
        # box = boxes[0, class, object detected]
        # Inside the array of the object detected:
        # index 1 is the class of the object
        # index 2 is the score
        # index 3 to 6 is the box coordinates
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]

        if score < 0.5:
            continue

        # If class is not person continue
        if class_id != 0:
            continue
        # Get box coordinates
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        roi = background[y: y2, x: x2]
        roi_height, roi_width, _ = roi.shape

        # Get the mask
        mask = masks[i, int(class_id)]

        mask = cv2.resize(mask, (roi_width, roi_height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # print(mask)

        cv2.rectangle(bgr_frame, (x, y), (x2, y2), (255, 0, 0), 2)

        # Get mask coordinated
        contours, _ = cv2.findContours(
            np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[int(class_id)]
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (int(color[0]),
                         int(color[1]), int(color[2])))

    # print(box)

    cv2.imshow("Image", bgr_frame)
    # cv2.imshow("Image", background)
    key = cv2.waitKey(1)
    if key == 27:
        break

rs.release()
cv2.destroyAllWindows()
