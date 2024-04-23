import numpy as np

# arr0 is a 2x2 image of [B, G, R, A] pixels
# where each pixel value represents magenta
arr0 = np.full((2, 2, 4), np.uint8(255))
arr0[:, :, 1] = np.uint8(0)

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

arr1 = pack_buffer(arr0)
arr2 = np.array([arr0.shape[1], arr0.shape[0]], dtype='float32').reshape(1, 2)
print(arr0)     # disp orig image
print(arr1)     # disp packed buffers
print(arr2)     # disp corr packed buffer sizes
