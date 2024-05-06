#!/usr/bin/env python3
import cv2
import depthai as dai
from depthai_sdk import Replay

# Create Replay object
replay = Replay("record/color.mp4")

if replay is None:
    raise Exception("Could not find file to replay!!")

# Initialize the pipeline. This will create required XLinkIn's and connect them together
pipeline, nodes = replay.initPipeline()

print(f"Node: {nodes}")

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("replay")
nodes.color.out.link(xout.input)

with dai.Device(pipeline) as device:
    replay.createQueues(device)
    q = device.getOutputQueue(name="replay", maxSize=4, blocking=False)

    while replay.sendFrames():
        frame = q.get().getCvFrame()
        # frame = q.get().getFrame()
        cv2.imshow("Replay Video", frame)
        if cv2.waitKey(1) == ord('q'):
            break
