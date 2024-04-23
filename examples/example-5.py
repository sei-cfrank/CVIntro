from libcamera import Transform
from picamera2 import Picamera2
from onnx2torch import convert

import cv2
import numpy as np

# letterbox procedure
def letterbox(src, dest_shape):
    # get src dims
    src_width = src.shape[1]    # img.shape returns tuple (rows, cols, chan)
    src_height = src.shape[0]   # NOTE: rows => height; cols => width

    # cons dest array (filled with gray), get dest dims
    # NOTE: each 32-bit [B, G, R, A] pixel value is [128, 128, 128, 255]
    dest = np.full(dest_shape, np.uint8(128))
    dest[:, :, 3] = np.uint8(255)
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

# pack_buffer procedure, ONNX model expects normalized float32 NCHW tensor
def pack_buffer(src):
    width = src.shape[1]
    height = src.shape[0]
    chan = src.shape[2]
    dest_shape = (chan - 1, height, width)      # cons dest array shape
    dest = np.array(src, dtype='float32')       # cons dest array via copy
    dest /= 255.0                               # normalize vals
    dest = np.transpose(dest, [2, 0, 1])        # make channel first dim
    dest = np.resize(dest, dest_shape)          # remove alpha channel
    dest = np.expand_dims(dest, 0)              # ins batch dim before chan dim
    return dest

# instantiate camera instance
picam2 = Picamera2()

# create a config with desired attributes: format, size, framerate, transform
# NOTE: camera resolution 3280x2464, downsamples at 820x616, crops at 640x480
# NOTE: XRGB8888 => shape: (height, width, 4); pixel value: [B, G, R, A]
config = picam2.create_preview_configuration(
    main={'format': 'XRGB8888', 'size': (820, 616)},
    controls={'FrameDurationLimits': (16667, 16667)},
    transform=Transform(hflip=True))

# set camera configuration, start camera
picam2.configure(config)
picam2.start()

# start opencv window thread
cv2.startWindowThread()
wnd_name = 'foo'
cv2.namedWindow(wnd_name, cv2.WINDOW_AUTOSIZE)

# cons PyTorch Tiny YOLOv3 NN using onnx2torch
onnx_model_path = '/path/to/tiny_yolo_v3.onnx'
tiny_yolo_v3 = convert(onnx_model_path)

while True:
    # get current image data from 'main' camera stream
    arr1 = picam2.capture_array('main')
    fps = 1000000 / picam2.capture_metadata()['FrameDuration']

    # letterbox the image to resize for NN input (size: (height, width, chan))
    arr2 = letterbox(arr1, (416, 416, 4))

    # cons packed input buffer for PyTorch model inference
    arr3 = pack_buffer(arr2)

    # run PyTorch model inference on input buffer to get results
    res = tiny_yolo_v3(arr3)

    # process results to make list of annotations
    # annos = proc_results(res)

    # draw list of annotations on letterboxed image
    # arr4 = draw_annotations(annos, arr1)

    # show annotated image
    # cv2.imshow(wnd_name, arr4)
    cv2.imshow(wnd_name, arr1)

    # show FPS in window title
    cv2.setWindowTitle(wnd_name, f'FPS: {fps:5.2f}')
