import numpy as np
import cv2
import depthai as dai
import imutils
from stitching import Stitcher

WIDTH, HEIGHT = (600, 444)

all_devices = dai.Device.getAllAvailableDevices()
print(f'Found {len(all_devices)} devices')

for dev in all_devices:
    id = dev.getMxId()
    print(f"Device id: {id}")

# /////////////////////// Device 1
pipeline1 = dai.Pipeline()
pipeline1.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

cam1 = pipeline1.create(dai.node.ColorCamera)
cam1.setPreviewSize(WIDTH, HEIGHT)
cam1.setInterleaved(False)
cam1.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam1.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

xout1 = pipeline1.create(dai.node.XLinkOut)
xout1.setStreamName("device1_output_to_host")

cam1.preview.link(xout1.input)
# cam1.isp.link(xout1.input)

info1 = dai.DeviceInfo("18443010D13E411300") # MXID
device1 = dai.Device(pipeline1, info1)
qDevice1 = device1.getOutputQueue(name="device1_output_to_host", maxSize=4, blocking=False)

# /////////////////////// Device 2
pipeline2 = dai.Pipeline()
pipeline2.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

cam2 = pipeline2.create(dai.node.ColorCamera)
cam2.setPreviewSize(WIDTH, HEIGHT)
cam2.setInterleaved(False)
cam2.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam2.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

xout2 = pipeline2.create(dai.node.XLinkOut)
xout2.setStreamName("device2_output_to_host")

cam2.preview.link(xout2.input)
# cam2.isp.link(xout2.input)

info2 = dai.DeviceInfo("18443010915D2D1300") # MXID
device2 = dai.Device(pipeline2, info2)
qDevice2 = device2.getOutputQueue(name="device2_output_to_host", maxSize=4, blocking=False)

# Get first images to compute sift/homography
img_cv1 = qDevice1.get().getCvFrame()
img_cv2 = qDevice2.get().getCvFrame()

stitcher = Stitcher([img_cv1, img_cv2])

translateX = 0
translateY = 0

# /////////////////////// Get images from devices and stitch them together
while True:
    # Load the images
    img_cv1 = qDevice1.get().getCvFrame()
    img_cv2 = qDevice2.get().getCvFrame()

    M = np.float32([
    [1, 0, translateX],
    [0, 1, translateY]
    ])

    if translateX != 0 or translateY != 0:
        img_cv2 = cv2.warpAffine(img_cv2, M, (WIDTH, HEIGHT))
    
    result = stitcher.warp([img_cv1, img_cv2])
    if result is not None:
        # print(f"Shape: {result.shape}")
        cv2.imshow("Concatenated images", result)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Quitting...")
        device1.close()
        device2.close()
        break
    elif key == ord('c'):
        stitcher = Stitcher([img_cv1, img_cv2])
    elif key == ord(']'):
        print("Translating to right...")
        translateX += 1
    elif key == ord('['):
        print("Translating to left...")
        translateX -= 1
    elif key == ord('u'):
        print("Translating to up...")
        translateY -= 1
    elif key == ord('d'):
        print("Translating to down...")
        translateY += 1