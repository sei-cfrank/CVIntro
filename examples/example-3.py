from libcamera import Transform
from picamera2 import Picamera2

import cv2
import numpy as np

# letterbox procedure
def letterbox(src, dest_shape):
    # get src dims
    src_width = src.shape[1]    # img.shape returns tuple (rows, cols, chan)
    src_height = src.shape[0]   # NOTE: rows => height; cols => width

    # cons dest array (filled with gray), get dest dims
    # NOTE: each 32-bit RGBA pixel value is 0x80808080, alpha is ignored 
    dest = np.full(dest_shape, np.uint8(128))
    dest_width = dest.shape[1]
    dest_height = dest.shape[0]

    # calculate width and height ratios
    width_ratio = dest_width / src_width        # NOTE: ratios are float values
    height_ratio = dest_height / src_height

    # init resized image width and height with max values (dest dims)
    rsz_width = dest_width
    rsz_height = dest_height

    # smallest scale factor will scale other dimension as well
    if width_ratio < height_ratio:
        rsz_height = int(src_height * width_ratio)  # NOTE: integer truncation
    else:
        rsz_width = int(src_width * height_ratio)

    # resize the image data using bi-linear interpolation
    rsz_dims = (rsz_width, rsz_height)
    rsz = cv2.resize(src, rsz_dims, 0, 0, cv2.INTER_LINEAR)

    # embed rsz into the center of dest
    dx = int((dest_width - rsz_width) / 2)          # NOTE: integer truncation
    dy = int((dest_height - rsz_height) / 2)
    dest[dy:dy+rsz_height, dx:dx+rsz_width, :] = rsz

    # letterboxing complete, return dest
    return dest


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

    # letterbox the image to resize for NN input 
    arr2 = letterbox(arr1, (512, 512, 4))

    # show resized image
    cv2.imshow('foo', arr2)
