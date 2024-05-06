import numpy as np
import cv2
import depthai as dai
import imutils
from stitching import Stitcher

all_devices = dai.Device.getAllAvailableDevices()
print(f'Found {len(all_devices)} devices')

for dev in all_devices:
    id = dev.getMxId()
    print(f"Device id: {id}")

# Create a pipeline on one device that outputs camera
# image data to host computer
# Create a pipeline on one device that takes input
# from host (other device) and then combines that with
# it's own image from camera

# /////////////////////// Device 1
pipeline1 = dai.Pipeline()
pipeline1.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

cam1 = pipeline1.create(dai.node.ColorCamera)
cam1.setPreviewSize(600, 444)
cam1.setInterleaved(False)
cam1.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam1.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

xout1 = pipeline1.create(dai.node.XLinkOut)
xout1.setStreamName("device1_out")

cam1.preview.link(xout1.input)

info1 = dai.DeviceInfo("18443010D13E411300") # MXID
device1 = dai.Device(pipeline1, info1)

qOut_device1 = device1.getOutputQueue(name="device1_out", maxSize=4, blocking=False)

# /////////////////////// Device 2
pipeline2 = dai.Pipeline()
pipeline2.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

warp = pipeline2.create(dai.node.Warp)
warp.setStreamName("warp")
warp.out.link(xout2.input)

xin2 = pipeline2.create(dai.node.XLinkIn)
xin2 = setStreamName("device2_in")
xin2.out.link(warp.input)

xout2 = pipeline2.create(dai.node.XLinkOut)
xout2.setStreamName("device2_out")

cam2 = pipeline2.create(dai.node.ColorCamera)
cam2.setPreviewSize(600, 444)
cam2.setInterleaved(False)
cam2.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam2.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

info2 = dai.DeviceInfo("18443010915D2D1300") # MXID
device2 = dai.Device(pipeline2, info2)

qIn_device2 = device2.getInputQueue(name="device2_in", maxSize=4, blocking=True)
qOut_device2 = device2.getOutputQueue(name="device2_out", maxSize=4, blocking=False)

# /////////////////////// Get images from devices and stitch them together
while True:
    # Get image frame from camera 1
    frame_dev1 = qOut_device1.get()

    
    frame_dev2 = qDevice2.get()
   
    # Load the images
    img_cv1 = frame_dev1.getCvFrame()
    img_cv2 = frame_dev2.getCvFrame()

    stitcher = Stitcher()
    result = stitcher.stitch([img_cv1, img_cv2], showMatches = False)
    
    if result is not None:
        print(f"Shape: {result.shape}")
        cv2.imshow("Concatenated images", result)
    
    if cv2.waitKey(1) == ord('q'):
        break
