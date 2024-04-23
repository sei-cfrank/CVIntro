from libcamera import Transform
from picamera2 import Picamera2

import cv2
import numpy as np

# instantiate camera instance
picam2 = Picamera2()

# create a config with desired attributes: format, size, framerate, transform
# NOTE: camera resolution 3280x2464, downsamples at 820x616, crops at 640x480
config = picam2.create_preview_configuration(
    main={'format': 'XRGB8888', 'size': (820, 616)},
    controls={'FrameDurationLimits': (16667, 16667)},
    transform=Transform(hflip=True))

# set camera configuration, start camera
picam2.configure(config)
picam2.start()

# start opencv window thread
cv2.startWindowThread()

while True:
    # get current image data from 'main' camera stream
    arr1 = picam2.capture_array('main')

    # resize the image data using bi-linear interpolation
    arr2 = cv2.resize(arr1, (640, 480), 0, 0, cv2.INTER_LINEAR)

    # show resized image
    cv2.imshow('foo', arr2)
