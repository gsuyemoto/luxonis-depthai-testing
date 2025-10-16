#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import numpy as np

def createPipeline(pipeline):
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    output = camRgb.requestOutput((1920,1080), dai.ImgFrame.Type.NV12 ,dai.ImgResizeMode.CROP, 20).createOutputQueue()
    return pipeline, output

with contextlib.ExitStack() as stack:
    deviceInfos = dai.Device.getAllAvailableDevices()
    usbSpeed = dai.UsbSpeed.SUPER
    openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4

    print(f"Number devices available: {len(deviceInfos)}")

    mxids = []
    queues = []
    pipelines = []

    for deviceInfo in deviceInfos:
        pipeline = stack.enter_context(dai.Pipeline())
        device = pipeline.getDefaultDevice()
        print("===Connected to ", deviceInfo.deviceId)
        mxId = device.getDeviceId()
        mxids.append(mxId)
        cameras = device.getConnectedCameras()
        usbSpeed = device.getUsbSpeed()
        eepromData = device.readCalibration2().getEepromData()
        print("   >>> MXID:", mxId)
        print("   >>> Num of cameras:", len(cameras))
        print("   >>> USB speed:", usbSpeed)
        if eepromData.boardName != "":
            print("   >>> Board name:", eepromData.boardName)
        if eepromData.productName != "":
            print("   >>> Product name:", eepromData.productName)

        # Define source and output
        pipeline, output = createPipeline(pipeline)
        pipeline.start()
        pipelines.append(pipeline)

        queues.append(output)

    while True:
        frames = []
        for i, stream in enumerate(queues):
            videoIn = stream.get()
            assert isinstance(videoIn, dai.ImgFrame)
            height = videoIn.getHeight()
            width = videoIn.getWidth()
            frame = videoIn.getData().reshape((height * 3 // 2, width)).astype(np.uint8)
            # Convert NV12 to BGR for display and concatenation
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
            # Resize frame to a common height (e.g., 360p) for display, maintaining aspect ratio
            display_height = 360
            display_width = int(frame_bgr.shape[1] * display_height / frame_bgr.shape[0])
            frame_resized = cv2.resize(frame_bgr, (display_width, display_height))
            frames.append(frame_resized)

        # for q_rgb, stream_name in qRgbMap:
        #     if stream_name == mxids[0]:
        #         frames[1] = q_rgb.get().getCvFrame()
        #     else:
        #         frames[0] = q_rgb.get().getCvFrame()
        
        if len(frames) > 0:
            # Concatenate frames horizontally
            combined_frame = cv2.hconcat(frames)
            
            # Display the concatenated frame
            cv2.imshow('Combined Stream', combined_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # while True:
    #         for i, stream in enumerate(queues):
    #             videoIn = stream.get()
    #             assert isinstance(videoIn, dai.ImgFrame)
    #             cv2.imshow(f"video_device{i}", videoIn.getCvFrame())
    #         if cv2.waitKey(1) == ord('q'):
    #             break
