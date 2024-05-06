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
xout1.setStreamName("device1_output_to_host")

cam1.preview.link(xout1.input)

info1 = dai.DeviceInfo("18443010D13E411300") # MXID
device1 = dai.Device(pipeline1, info1)

qDevice1 = device1.getOutputQueue(name="device1_output_to_host", maxSize=4, blocking=False)

# /////////////////////// Device 2
pipeline2 = dai.Pipeline()
pipeline2.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

cam2 = pipeline2.create(dai.node.ColorCamera)
cam2.setPreviewSize(600, 444)
cam2.setInterleaved(False)
cam2.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam2.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

xout2 = pipeline2.create(dai.node.XLinkOut)
xout2.setStreamName("device2_output_to_host")

xin2 = pipeline2.create(dai.node.XLinkIn)
xin2 = setStreamName("device2_input_from_host")

cam2.preview.link(xout2.input)

info2 = dai.DeviceInfo("18443010915D2D1300") # MXID
device2 = dai.Device(pipeline2, info2)

qIn_device2 = device2.getInputQueue(name="device2_input_from_host", maxSize=4, blocking=True)
qOut_device2 = device2.getOutputQueue(name="device2_output_to_host", maxSize=4, blocking=False)

# /////////////////////// Get images from devices and stitch them together
while True:
    # images_to_be_stitched = []
    
    frame_dev1 = qDevice1.get()
    frame_dev2 = qDevice2.get()
   
    # Load the images
    img_cv1 = frame_dev1.getCvFrame()
    img_cv2 = frame_dev2.getCvFrame()

    # # Convert to grayscale
    # img_cv1 = cv.cvtColor(img_cv1,cv.COLOR_BGR2GRAY)
    # img_cv2 = cv.cvtColor(img_cv2,cv.COLOR_BGR2GRAY)
    
    # # Detect and match features
    # sift = cv2.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(img_cv1, None)
    # kp2, des2 = sift.detectAndCompute(img_cv2, None)
    # matches = bf.match(des1, des2)
    
    # # Compute the homography matrix
    # H, _ = cv2.findHomography(kp1, kp2, matches)
    
    # # Warp the second image
    # warped_img = cv2.warpPerspective(img_cv2, H, (img_cv1.shape[1], img_cv1.shape[0]))

    # numpy_horizontal_concat = np.concatenate((img_cv1, img_cv2), axis=1)
    # cv2.imshow("Concatenated images", numpy_horizontal_concat)

    stitcher = Stitcher()
    result = stitcher.stitch([img_cv1, img_cv2], showMatches = False)
    
    if result is not None:
        print(f"Shape: {result.shape}")
        cv2.imshow("Concatenated images", result)
    
    if cv2.waitKey(1) == ord('q'):
        break
