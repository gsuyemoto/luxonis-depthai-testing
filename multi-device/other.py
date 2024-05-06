import numpy as np
import cv2
import depthai as dai

SHAPE = 300

# Create a pipeline on one device that outputs camera
# image data to host computer
# Create a pipeline on one device that takes input
# from host (other device) and then combines that with
# it's own image from camera

# /////////////////////// Device 1 to Host
pipeline1 = dai.Pipeline()

xout = pipeline1.createXLinkOut()
xout.setStreamName("device1_output_to_host")

cam1 = p.create(dai.node.ColorCamera)
cam1.setPreviewSize(SHAPE, SHAPE)
cam1.setInterleaved(False)
cam1.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# /////////////////////// Device 2 input from Host
pipeline2 = dai.Pipeline()
pipeline2.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

xin = pipeline2.createXLinkIn()
xin.setStreamName("device2_input_from_host")

all_devices = dai.Device.getAllAvailableDevices()
print(f'Found {len(device_infos)} devices')

cam2 = pipeline2.create(dai.node.ColorCamera)
cam2.setPreviewSize(SHAPE, SHAPE)
cam2.setInterleaved(False)
cam2.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# NN that concats images
nn = p.create(dai.node.NeuralNetwork)
nn.setBlobPath("models/concat_openvino_2021.4_6shave.blob")
nn.setNumInferenceThreads(2)

manipLeft.out.link(nn.inputs['img1'])
camRgb.preview.link(nn.inputs['img2'])

# Send bouding box from the NN to the host via XLink
nn_xout = p.create(dai.node.XLinkOut)
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    shape = (3, SHAPE, SHAPE * 3)

    while True:
        inNn = np.array(qNn.get().getData())
        frame = inNn.view(np.float16).reshape(shape).transpose(1, 2, 0).astype(np.uint8).copy()

        cv2.imshow("Concat", frame)

        if cv2.waitKey(1) == ord('q'):
            break
