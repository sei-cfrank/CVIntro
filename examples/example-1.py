from picamera2 import Picamera2
import time

# instantiate camera instance
picam2 = Picamera2()

# create a config with desired attributes: format, size, framerate
# NOTE: camera resolution 4608x2464, downsamples at 2304x1296 (56.03 fps)
# NOTE: XRGB8888 => shape: (height, width, 4); pixel value: [B, G, R, A]
config = picam2.create_preview_configuration(
    main={'format': 'XRGB8888', 'size': (2304, 1296)})  # 16:9 aspect ratio

# set camera configuration, print config to show results
picam2.configure(config)
print(config)

# start camera with preview, set preview title fields, and sleep for 8 seconds
picam2.start(show_preview=True)
picam2.title_fields = ['FrameDuration']
time.sleep(8)

# capture image data as np-array with XBGR8888 format, print np-array shape
array = picam2.capture_array('main')
print(array.shape)
