#!/usr/bin/env python3

import cv2
import depthai as dai
import contextlib
import numpy as np

def createPipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    # Define a source - color camera
    camRgb = pipeline.create(dai.node.ColorCamera)

    camRgb.setPreviewSize(300, 300)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)

    # Create output
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    return pipeline


with contextlib.ExitStack() as stack:
    deviceInfos = dai.Device.getAllAvailableDevices()
    usbSpeed = dai.UsbSpeed.SUPER
    openVinoVersion = dai.OpenVINO.Version.VERSION_2021_4

    qRgbMap = []
    devices = []
    mxids = []

    for deviceInfo in deviceInfos:
        deviceInfo: dai.DeviceInfo
        device: dai.Device = stack.enter_context(dai.Device(openVinoVersion, deviceInfo, usbSpeed))
        devices.append(device)
        print("===Connected to ", deviceInfo.getMxId())
        mxId = device.getMxId()
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

        pipeline = createPipeline()
        device.startPipeline(pipeline)

        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qRgbMap.append((q_rgb, mxId))

    while True:
        frames = [None, None]
        for q_rgb, stream_name in qRgbMap:
            if stream_name == mxids[0]:
                frames[1] = q_rgb.get().getCvFrame()
            else:
                frames[0] = q_rgb.get().getCvFrame()
        
        if len(frames) > 0:
            # Concatenate frames horizontally
            combined_frame = cv2.hconcat(frames)
            
            # Display the concatenated frame
            cv2.imshow('Combined Stream', combined_frame)

        if cv2.waitKey(1) == ord('q'):
            break
