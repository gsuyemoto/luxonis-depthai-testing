import numpy as np
import cv2
import depthai as dai
import imutils

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

cam2.preview.link(xout2.input)

info2 = dai.DeviceInfo("18443010915D2D1300") # MXID
device2 = dai.Device(pipeline2, info2)
qDevice2 = device2.getOutputQueue(name="device2_output_to_host", maxSize=4, blocking=False)

# /////////////////////// Get images from devices and stitch them together
while True:
    # images_to_be_stitched = []
    
    frame_dev1 = qDevice1.get()
    # images_to_be_stitched.append(frame_dev1.getCvFrame())
    
    frame_dev2 = qDevice2.get()
    # images_to_be_stitched.append(frame_dev2.getCvFrame())
    
    # if imutils.is_cv3():
    #     print("Using CV3 createStitcher...")
    #     stitcher = cv2.createStitcher()
    # else:
    #     print("Using CV4 Stitcher_create...")
    #     stitcher = cv2.Stitcher_create()

    # (status, stitched) = stitcher.stitch(images_to_be_stitched)

    # if status == 0:
    #     cv2.imshow("Stitched from 2 cameras", stitched)
    # else:
    #     print("Unable to stitch images...")
    
    numpy_horizontal_concat = np.concatenate((frame_dev1.getCvFrame(), frame_dev2.getCvFrame()), axis=1)
    cv2.imshow("Concatenated images", numpy_horizontal_concat)
    
    if cv2.waitKey(1) == ord('q'):
        break
