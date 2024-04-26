from picamera2 import Picamera2
import time

# instantiate camera instance
picam2 = Picamera2()

# create a config with desired attributes: format, size, framerate
config = picam2.create_preview_configuration(
    main={'format': 'XBGR8888', 'size': (1640, 1232)},
    controls={'FrameDurationLimits': (16667, 16667)})

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
