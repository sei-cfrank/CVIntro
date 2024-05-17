## Notes on Examples

### Example 0
Instantiates picam object and starts preview window

### Example 1
Create a config with desired attributes: format, size, framerate
    NOTE: camera resolution 4608x2464, downsamples at 2304x1296 (56.03 fps)
    NOTE: XRGB8888 => shape: (height, width, 4); pixel value: [B, G, R, A]
Start camera with preview, set preview title fields, and sleep for 8 seconds
Capture image data as np-array with XBGR8888 format, print np-array shape

### Example 2
''
Start opencv WindowThread
Get current image data from 'main' camera stream
Resize the image data using bi-linear interpolation
Show resized image in cv WindowThread

### Example 3
''
Same as above except that we letterbox the array before showing it in the cv WindowThread

### Example 4
''
pack_buffer procedure, ONNX model expects normalized float32 NCHW tensor
we pack buffer after letterboxing, this involves
    remove alpha channel
    reorder channels: BGR -> RGB
    normalize vals by /255 (Ask aren't pixel values 128??)
    make channel first dim
    ins batch dim before chan dim

### Example 5
''
Instantiate TinyYolo model for inference
Run inference on the letterboxed and buffer packed image


### Example 6
''
Process the results from inference to make a list of annotations
`annos = proc_results(res)`
Draw the annotations on the letterboxed image:
`arr4 = draw_annos(arr2, annos)`
Display annotated image

### Example 7
''
Unscale annotations to draw in original image frame
Draw list of annotations on original image

### Example 8
''
New proc_results function for more control
`def proc_results(res, pobj_thresh = 0.1, pcls_thresh = 0.5,...`
New functions for overlap, iou, basic NMS, make_annos
