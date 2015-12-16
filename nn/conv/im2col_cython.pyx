import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def im2col_cython(np.ndarray[DTYPE_t, ndim=4] x,
                  int kernel_height,
                  int kernel_width,
                  int padding,
                  int stride):
    """ This function takes in am image x and cuts out all kernel_height x kernel_width
    sub images from it, respecting the provided padding and stride variables.
    These small sub images are then placed into columns of a new matrix, 
    which will be returned.
    """
    # we assume that the shape of x is:
    # (batch_size, channels, height, width)
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]

    # calculate output sizes
    cdef int H_OUT = (H + 2 * padding - kernel_height) / stride + 1
    cdef int W_OUT = (W + 2 * padding - kernel_width) / stride + 1

    cdef int p = padding
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.pad(x,
            ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    cdef np.ndarray[DTYPE_t, ndim=2] cols = np.zeros(
            (C * kernel_height * kernel_width, N * H_OUT * W_OUT), dtype=DTYPE)


    im2col_loop(cols, x_padded, N, C, H, W, H_OUT, W_OUT,
                kernel_height, kernel_width, padding, stride)

    return cols

@cython.boundscheck(False)
cdef int im2col_loop(np.ndarray[DTYPE_t, ndim=2] cols,
                     np.ndarray[DTYPE_t, ndim=4] x_padded,
                     int N, int C, int H, int W, int H_OUT, int W_OUT,
                     int kernel_height, int kernel_width, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for ii in range(kernel_height):
            for jj in range(kernel_width):
                row = c * kernel_width * kernel_height + ii * kernel_height + jj
                for yy in range(H_OUT):
                    for xx in range(W_OUT):
                        for i in range(N):
                            col = yy * W_OUT * N + xx * N + i
                            cols[row, col] = x_padded[i, c, stride * yy + ii, stride * xx + jj]


def col2im_cython(np.ndarray[DTYPE_t, ndim=2] cols, int N, int C, int H, int W,
                  int kernel_height, int kernel_width, int padding, int stride):
    """ This function is the reverse of im2col. It takes in a matrix cols in which the columns
    are assumed to be cut outs of size kernel_width x kernel_height of an original image x.
    It then assembles these small images into one bigger image of H x W again.
    """
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=DTYPE)
    cdef int H_OUT = (H + 2 * padding - kernel_height) / stride + 1
    cdef int W_OUT = (W + 2 * padding - kernel_width) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding),
                                        dtype=DTYPE)

    col2im_loop(cols, x_padded, N, C, H, W, H_OUT, W_OUT, 
                kernel_height, kernel_width, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

@cython.boundscheck(False)
cdef int col2im_loop(np.ndarray[DTYPE_t, ndim=2] cols,
                     np.ndarray[DTYPE_t, ndim=4] x_padded,
                     int N, int C, int H, int W, int H_OUT, int W_OUT,
                     int kernel_height, int kernel_width, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for ii in range(kernel_height):
            for jj in range(kernel_width):
                row = c * kernel_width * kernel_height + ii * kernel_height + jj
                for yy in range(H_OUT):
                    for xx in range(W_OUT):
                        for i in range(N):
                            col = yy * W_OUT * N + xx * N + i
                            x_padded[i, c, stride * yy + ii, stride * xx + jj] += cols[row, col]
