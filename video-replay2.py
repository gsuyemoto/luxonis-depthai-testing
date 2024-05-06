import cv2
import depthai as dai
from depthai_sdk import Replay

# Get all devices
all_devices = dai.Device.getAllAvailableDevices()

# Create Replay objects
# replay = Replay(args.path)
replay = Replay("color.mp4")

if replay is None:
    raise Exception("Could not find file to replay!!")

replay.disableStream('depth') # In case depth was saved (mcap)
# Resize color frames prior to sending them to the device
replay.setResizeColor((304, 304))
# Keep aspect ratio when resizing the color frames. This will crop
# the color frame to the desired aspect ratio (in our case 300x300)
replay.keepAspectRatio(True)

# Initializes the pipeline. This will create required XLinkIn's and connect them together
# Creates StereoDepth node, if both left and right streams are recorded
pipeline = replay.initPipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(1000, 1000)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras:', device.getConnectedCameraFeatures())
    # Print out usb speed
    print('Usb speed:', device.getUsbSpeed().name)
    # Bootloader version
    if device.getBootloaderVersion() is not None:
        print('Bootloader version:', device.getBootloaderVersion())
    # Device name
    print('Device name:', device.getDeviceName(), ' Product name:', device.getProductName())

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

        # Retrieve 'bgr' (opencv format) frame
        cv2.imshow("rgb", inRgb.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
