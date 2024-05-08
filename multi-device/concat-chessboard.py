import numpy as np
import cv2
import depthai as dai
import imutils

CHESSBOARD_SIZE = (10, 7) # number of squares on the checkerboard
SQUARE_SIZE = 0.0235 # size of a square in meters
WIDTH = 600
HEIGHT = 444

def getHomography(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    print("Finding checkerboard corners...")
    
    chessboard_inner_size = (CHESSBOARD_SIZE[0] - 1, CHESSBOARD_SIZE[1] - 1)
    corners_world = np.zeros((1, chessboard_inner_size[0] * chessboard_inner_size[1], 3), np.float32)
    corners_world[0,:,:2] = np.mgrid[0:chessboard_inner_size[0], 0:chessboard_inner_size[1]].T.reshape(-1, 2)
    corners_world *= SQUARE_SIZE
    
    found1, corners1 = cv2.findChessboardCorners(
        img1_gray, chessboard_inner_size, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    found2, corners2 = cv2.findChessboardCorners(
        img2_gray, chessboard_inner_size, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if not found1:
        raise Exception("Chessboard on camera 1 not found")
    
    if not found2:
        raise Exception("Chessboard on camera 2 not found")
    
    # refine the corner locations
    corners1 = cv2.cornerSubPix(
        img1_gray, corners1, (11, 11), (-1, -1), 
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    
    corners2 = cv2.cornerSubPix(
        img2_gray, corners2, (11, 11), (-1, -1), 
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    
    # Get homography
    ransacMethod = cv2.RANSAC
    
    if cv2.__version__ >= "4.5.4":
        print("Using USAC_MAGSAC...")
        ransacMethod = cv2.USAC_MAGSAC
        
    homography, mask = cv2.findHomography(corners1, corners2, method = ransacMethod, ransacReprojThreshold = 5.0)

    print("Returning homography...")
    return homography

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

info2 = dai.DeviceInfo("18443010915D2D1300") # MXID
device2 = dai.Device(pipeline2, info2)
qDevice2 = device2.getOutputQueue(name="device2_output_to_host", maxSize=4, blocking=False)

homography = None

translateX = 0
translateY = 0

# /////////////////////// Get images from devices and stitch them together
while True:
    # Load the images
    imageA = qDevice1.get().getCvFrame()
    imageB = qDevice2.get().getCvFrame()

    M = np.float32([
    [1, 0, translateX],
    [0, 1, translateY]
    ])

    if translateX != 0 or translateY != 0:
        # imageB = cv2.warpAffine(imageB, M, (imageB.shape[1], imageB.shape[0]))
        imageB = cv2.warpAffine(imageB, M, (WIDTH, HEIGHT))
    
    if homography is not None:
        # print("Warping...")
        # imageB = cv2.warpPerspective(imageA, homography, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        imageB = cv2.warpPerspective(imageB, homography, (WIDTH, HEIGHT))

    concat_images = np.concatenate((imageA, imageB), axis=1)
    cv2.imshow("Two Cameras", concat_images)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Quitting...")
        device1.close()
        device2.close()
        break
    elif key == ord('c'):
        print("Getting homography...")
        homography = getHomography(imageA, imageB)
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
    elif key == ord('r'):
        print("Reset...")
        homography = None
