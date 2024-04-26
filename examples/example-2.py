from picamera2 import Picamera2

import cv2

# instantiate camera instance
picam2 = Picamera2()

# create a config with desired attributes: format, size, framerate
# NOTE: camera resolution 4608x2464, downsamples at 1536x864 (120.13 fps)
# NOTE: XRGB8888 => shape: (height, width, 4); pixel value: [B, G, R, A]
config = picam2.create_preview_configuration(
    main={'format': 'XRGB8888', 'size': (1536, 864)},
    controls={'FrameDurationLimits': (8333, 8333)})

# set camera configuration, start camera
picam2.configure(config)
picam2.start()

# start opencv window thread
cv2.startWindowThread()
wnd_name = 'foo'
cv2.namedWindow(wnd_name, cv2.WINDOW_AUTOSIZE)

while True:
    # get current image data from 'main' camera stream
    arr1 = picam2.capture_array('main')

    # resize the image data using bi-linear interpolation
    arr2 = cv2.resize(arr1, (640, 480), 0, 0, cv2.INTER_LINEAR)

    # show resized image
    cv2.imshow(wnd_name, arr2)
    cv2.waitKey(1)
