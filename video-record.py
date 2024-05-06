from depthai_sdk import OakCamera, RecordType
import time

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='4K', fps=30, encode='H265')

    # Sync & save all (encoded) streams
    oak.record([color.out.encoded], './record')
    oak.start()
    start_time = time.monotonic()
    while oak.running():
        if time.monotonic() - start_time > 5:
            break
        oak.poll()