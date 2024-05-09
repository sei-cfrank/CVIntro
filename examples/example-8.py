from picamera2 import Picamera2

import cv2
import math
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
    scale = 0.5
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

#+BEGIN_EXAMPLE

# sigmoid procedure
def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))

# (redefined) proc_results procedure
def proc_results(res, pobj_thresh = 0.1, pcls_thresh = 0.5, orig_img_size = 416,
                 anchors = np.array([[[81,82], [135,169], [344,319]],
                                     [[10,14], [ 23, 27], [ 37, 58]]],
                                    dtype='int32')):
    dets = []
    # candidate detection layout:
    # [x, y, w, h, pobj, pcls_0, pcls_1, ..., pcls_i]
    # i: [0, num_classes)
    num_classes = len(coco_names)
    pcls_offset = 5                                     # offset of class probs
    num_params = pcls_offset + num_classes              # numParams per cand det
    num_yolo_blocks = anchors.shape[0]
    num_anchors = anchors.shape[1]
    assert len(res) == num_yolo_blocks
    for blk in range(num_yolo_blocks):                  # iter over yolo blocks
        height_blk = res[blk].shape[1]
        width_blk = res[blk].shape[2]
        stride_blk = orig_img_size / width_blk          # ASSUMES square image
        shape_blk = (height_blk, width_blk, num_anchors, num_params)
        dets_blk = np.reshape(res[blk], shape_blk)
        # each yolo block has an "image" where each "pixel" has a candidate
        # detection per anchor box
        for hi in range(height_blk):                    # iter over img rows
            for wi in range(width_blk):                 # iter over img cols
                for ai in range(num_anchors):           # iter over pxl anchors
                    det = dets_blk[hi][wi][ai]          # get detection
                    pobj = sigmoid(det[4])              # get objectness prob
                    if pobj > pobj_thresh:
                        x = stride_blk * (wi + sigmoid(det[0]))
                        y = stride_blk * (hi + sigmoid(det[1]))
                        w = math.exp(det[2]) * anchors[blk][ai][0]
                        h = math.exp(det[3]) * anchors[blk][ai][1]
                        for ci in range(num_classes):
                            pcls = sigmoid(det[pcls_offset + ci])
                            if pcls > pcls_thresh:
                                x1, y1 = x - (w / 2.0), y - (h / 2.0)
                                x2, y2 = x + (w / 2.0), y + (h / 2.0)
                                dets.append((pobj, pcls, ci, x1, y1, x2, y2))
    return dets

# overlap procedure, find bbox overlap length along a dim
def overlap(lo1, hi1, lo2, hi2):
    lo = max(lo1, lo2)
    hi = min(hi1, hi2)
    return hi - lo

# iou procedure (intersection-over-union); bbox: [xl, yl, xh, yh]
def iou(bbox1, bbox2):
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])   # bbox1 area
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])   # bbox2 area
    wo = overlap(bbox1[0], bbox1[2], bbox2[0], bbox2[2])    # overlap x dim
    ho = overlap(bbox1[1], bbox1[3], bbox2[1], bbox2[3])    # overlap y dim
    i_area = (wo * ho) if (wo > 0.0 and ho > 0.0) else 0.0  # intersection area
    u_area = area1 + area2 - i_area                         # union area
    return i_area / u_area

# basic_nms procedure (non-maximum supression); det: (pobj,pcls,ci,x1,y1,x2,y2)
def basic_nms(dets, iou_thresh = 0.5):
    filtered_dets = []
    dets.sort(reverse=True)                     # lexicographically sort dets
    while len(dets) > 0:                        # any remaining dets to check?
        c = dets[0]                             # get current det
        filtered_dets.append(c)                 # add to filtered_dets
        # predicate remove dets with same class index and high iou
        pred = lambda d : not (c[2] == d[2] and iou(c[3:], d[3:]) > iou_thresh)
        dets = [d for d in dets if pred(d)]     # make list of remaining dets
    return filtered_dets

# make_annos procedure
def make_annos(dets):
    annos = []
    for det in dets:
        box = [det[4], det[3], det[6], det[5]]  # NOTE: replicate ONNX bug
        score = det[0] * det[1]
        cls = det[2]
        annos.append((box, score, cls))
    return annos

#+END_EXAMPLE

# cons ONNX Tiny YOLOv3 NN model
onnx_model_path = '../model/modified_yolov3-tiny.onnx'
infer_sess = ort.InferenceSession(onnx_model_path)

# instantiate camera instance
picam2 = Picamera2()

# create a config with desired attributes: format, size, framerate
# NOTE: camera resolution 4608x2464, downsamples at 2304x1296 (56.03 fps)
# NOTE: XRGB8888 => shape: (height, width, 4); pixel value: [B, G, R, A]
config = picam2.create_preview_configuration(
    main={'format': 'XRGB8888', 'size': (2304, 1296)})  # 16:9 aspect ratio

# set camera configuration, start camera
picam2.configure(config)
picam2.start()

# start opencv window thread
cv2.startWindowThread()
wnd_name = 'foo'
cv2.namedWindow(wnd_name, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(wnd_name, 1600, 900)                   # 16:9 aspect ratio
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

    # run ONNX model inference on input buffer to get results
    res = infer_sess.run(None, {'input_1': arr3})

    # process results to make list of annotations
    # TODO: change pobj_thresh, pcls_thresh, and iou_thresh
    dets = proc_results(res, 0.1, 0.5)
    filtered_dets = basic_nms(dets, 0.5)
    filtered_annos = make_annos(filtered_dets)

    # unscale annotations to draw in original image frame
    unscaled = unscale_annos(filtered_annos, dw, dh, w0, h0, w1, h1)

    # draw list of annotations on original image
    arr4 = draw_annos(arr1, unscaled)

    # update fps timer
    t1 = timeit.default_timer()
    fps = 1.0 / (t1 - t0)

    # if window closed, break loop before imshow creates new window
    if cv2.getWindowProperty(wnd_name, cv2.WND_PROP_AUTOSIZE) == -1:
        break

    # show annotated image
    cv2.setWindowTitle(wnd_name, f'FPS: {fps:.1f}')
    cv2.imshow(wnd_name, arr4)
    cv2.waitKey(1)
