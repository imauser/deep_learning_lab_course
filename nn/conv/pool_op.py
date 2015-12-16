import numpy as np
from im2col_cython import col2im_cython, im2col_cython


def forward_max_pool(imgs, pool_height, pool_width, stride):
    N, C, H, W = imgs.shape
    # DO some sanity checking on dimensions
    if (W - pool_width) % stride != 0:
        raise ValueError("Pool width {} does not work for image width {}".format(pool_width, W))
    if (H - pool_height) % stride != 0:
        raise ValueError("Pool height {} does not work for image height {}".format(pool_height, H))
    out_height = (H - pool_height) / stride + 1
    out_width = (W - pool_width) / stride + 1

    # mangle toegther batch size and channels for all images
    # this way we can simply do an argmax after calling im2col
    imgs_cols = im2col_cython(imgs.reshape(N*C, 1, H, W), pool_height, pool_width, 0, stride)
    # compute maximum
    imgs_argmax = np.argmax(imgs_cols, axis=0)
    # get maximum values
    imgs_max = imgs_cols[imgs_argmax, np.arange(imgs_cols.shape[1])]
    # the output will simply be the selected maximum values reshaped
    # we have to do some transposes here since the maxima are currently
    # in im2col format
    res = imgs_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
    return res, imgs_cols, imgs_argmax


def bprop_max_pool(imgs,
                   imgs_cols,
                   imgs_argmax,
                   poolout_grad,
                   pool_height, pool_width, stride):
    N, C, H, W = imgs.shape
    
    # first reorder and flatten the output gradient
    # so that we can simply extract the gradient at
    # maximum positions
    poolout_grad_reshaped = poolout_grad.transpose(2, 3, 0, 1).flatten()
    dImg_cols = np.zeros_like(imgs_cols)
    dImg_cols[imgs_argmax, np.arange(imgs_cols.shape[1])] = poolout_grad_reshaped
    dImg = col2im_cython(dImg_cols, N * C, 1, H, W, pool_height, pool_width, 0, stride)

    return dImg.reshape(imgs.shape)
