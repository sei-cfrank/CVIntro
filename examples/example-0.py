from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL, x=100, y=200, width=640, height=480)
picam2.start()
time.sleep(8)
