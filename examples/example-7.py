from picamera2 import Picamera2

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import timeit

# COCO class names
coco_file = open('../model/coco.names','r')
coco_names = []
while True:
    class_name = coco_file.readline().strip()
    if not class_name:
        break
    coco_names.append(class_name)
coco_file.close()

# letterbox procedure
def letterbox(src, dest_shape):
    # get src dims
    src_width = src.shape[1]    # array.shape returns (rows, cols, chan)
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
    rsz_origin = (dx, dy)

    # letterboxing complete, return (dest, rsz_origin, rsz_dims)
    return (dest, rsz_origin, rsz_dims)

# pack_buffer procedure, ONNX model expects normalized float32 NCHW tensor
def pack_buffer(src):
    dest = np.array(src, dtype='float32')       # cons dest array via copy
    dest = dest[:, :, :3]                       # remove alpha channel
    dest = dest[..., ::-1]                      # reorder channels: BGR -> RGB
    dest /= 255.0                               # normalize vals
    dest = np.transpose(dest, [2, 0, 1])        # make channel first dim
    dest = np.expand_dims(dest, 0)              # ins batch dim before chan dim
    return dest

# proc_results procedure
def proc_results(res):
    [boxes, scores, indices] = res
    out_boxes, out_scores, out_classes = [], [], []
    for idx in indices[0]:
        out_classes.append(idx[1])
        out_scores.append(scores[tuple(idx)])
        idx1 = (idx[0], idx[2])
        out_boxes.append(boxes[idx1])
    return list(zip(out_boxes, out_scores, out_classes))

# draw_annos procedure (fixed ONNX anno scaling in unscale_annos proc)
def draw_annos(src, annos):
    dest = np.copy(src)
    green = (0, 255, 0)
    black = (0, 0, 0)
    face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1
    for anno in annos:
        pt1 = (anno[0][0], anno[0][1])
        pt2 = (anno[0][2], anno[0][3])
        text = f'{coco_names[anno[2]]}: {anno[1]:.2f}'
        (w, h), _ = cv2.getTextSize(text, face, scale, thickness)
        pt3 = (pt1[0], pt1[1] - h)
        pt4 = (pt1[0] + w, pt1[1])
        dest = cv2.rectangle(src, pt1, pt2, green)
        dest = cv2.rectangle(dest, pt3, pt4, green, cv2.FILLED)
        dest = cv2.putText(dest, text, pt1, face, scale, black, thickness)
    return dest

# unscale_annos procedure (fixes ONNX anno scaling)
def unscale_annos(annos, dw, dh, w0, h0, w1, h1):
    res = []
    scale_w = float(w1) / float(w0)
    scale_h = float(h1) / float(h0)
    for anno in annos:
        pt1 = (int(anno[0][1]), int(anno[0][0]))   # ONNX bug! Points are
        pt2 = (int(anno[0][3]), int(anno[0][2]))   # transposed.
        pt3 = (pt1[0] - dw, pt1[1] - dh)
        pt4 = (pt2[0] - dw, pt2[1] - dh)
        pt5 = (int(float(pt3[0]) * scale_w), int(float(pt3[1]) * scale_h))
        pt6 = (int(float(pt4[0]) * scale_w), int(float(pt4[1]) * scale_h))
        arr1 = np.array([pt5[0], pt5[1], pt6[0], pt6[1]], dtype='int32')
        res.append((arr1, anno[1], anno[2]))
    return res

# cons ONNX Tiny YOLOv3 NN model
onnx_model_path = '../model/yolov3-tiny.onnx'
infer_sess = ort.InferenceSession(onnx_model_path)

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
t0 = timeit.default_timer()
t1 = timeit.default_timer()

while True:
    # update old time
    t0 = t1

    # get current image data from 'main' camera stream
    arr1 = picam2.capture_array('main')
    (h1, w1, c1) = arr1.shape

    # letterbox the image to resize for NN input (size: (height, width, chan))
    (arr2, (dw, dh), (w0, h0)) = letterbox(arr1, (416, 416, 4))

    # cons packed input buffer and dims for ONNX model inference
    arr3 = pack_buffer(arr2)
    dim3 = np.array([arr2.shape[1],arr2.shape[0]],dtype=np.float32).reshape(1,2)

    # run ONNX model inference on input buffer to get results
    res = infer_sess.run(None, {'input_1': arr3, 'image_shape': dim3})

    # process results to make list of annotations
    annos = proc_results(res)

    # unscale annotations to draw in original image frame
    unscaled = unscale_annos(annos, dw, dh, w0, h0, w1, h1)

    # draw list of annotations on letterboxed image
    arr4 = draw_annos(arr1, unscaled)

    # show annotated image
    t1 = timeit.default_timer()
    fps = 1.0 / (t1 - t0)
    cv2.setWindowTitle(wnd_name, f'FPS: {fps:.1f}')
    cv2.imshow(wnd_name, arr4)
    cv2.waitKey(1)
