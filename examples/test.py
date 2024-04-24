import cv2
import numpy as np
import onnx
import onnxruntime as ort
import time

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
    src_width = src.shape[1]    # img.shape returns tuple (rows, cols, chan)
    src_height = src.shape[0]   # NOTE: rows => height; cols => width

    # cons dest array (filled with gray), get dest dims
    # NOTE: each 32-bit [R, G, B] pixel value is [128, 128, 128]
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

# pack_buffer procedure, ONNX model expects normalized float32 NCHW tensor
def pack_buffer(src):
    dest = np.array(src, dtype='float32')       # cons dest array via copy
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

# draw_annos procedure
def draw_annos(src, annos):
    dest = np.copy(src)
    print(f'>>> annos\n{annos}')
    green = (0, 255, 0)
    black = (0, 0, 0)
    face = cv2.FONT_HERSHEY_TRIPLEX
    scale = 0.5
    thickness = 1
    for anno in annos:
        pt1 = (int(anno[0][0]), int(anno[0][1]))
        pt2 = (int(anno[0][2]), int(anno[0][3]))
        text = f'{coco_names[anno[2]]}: {anno[1]:6.4f}'
        (w, h), _ = cv2.getTextSize(text, face, scale, thickness)
        pt3 = (pt1[0], int(pt1[1] - h))
        pt4 = (int(pt1[0] + w), pt1[1])
        dest = cv2.rectangle(src, pt1, pt2, green)
        dest = cv2.rectangle(dest, pt3, pt4, green, cv2.FILLED)
        dest = cv2.putText(dest, text, pt1, face, scale, black, thickness)
    return dest

# cons ONNX Tiny YOLOv3 NN model
onnx_model_path = '../model/yolov3-tiny.onnx'
infer_sess = ort.InferenceSession(onnx_model_path)

# start opencv window thread
cv2.startWindowThread()
wnd_name = 'foo'
cv2.namedWindow(wnd_name, cv2.WINDOW_AUTOSIZE)

# open test image
arr1 = cv2.imread('../data/eagle.jpg')  # default: bgr for display
arr2 = arr1[..., ::-1]                  # bgr -> rgb for inference

# letterbox the image to resize for NN input (size: (height, width, chan))
arr3 = letterbox(arr2, (416, 416, 3))

# cons input for ONNX model inference (packed images and their orig dims)
arr4 = pack_buffer(arr3)
dim4 = np.array([arr1.shape[1], arr1.shape[0]], dtype=np.float32).reshape(1, 2)

# run ONNX model inference on input buffer to get results
res = infer_sess.run(None, {'input_1': arr4, 'image_shape': dim4})

# process results to make list of annotations
annos = proc_results(res)

# draw list of annotations on letterboxed image
arr5 = draw_annos(arr1, annos)

# show annotated image
cv2.imshow(wnd_name, arr5)
time.sleep(5)
