import numpy as np
from im2col_cython import col2im_cython, im2col_cython


def forward_conv(imgs, filters, stride, pad):
    N, C, H, W = imgs.shape
    # flip filters suhc that the number of outputs is
    # on the first dimension, this makes dot products
    # with the column matrix we get from im2col easier
    w = np.transpose(filters, axes=(1,0,2,3))
    num_filters, _, filter_height, filter_width = w.shape
    # DO some sanity checking on dimensions
    if (W + 2 * pad - filter_width) % stride != 0:
        raise ValueError("Filter width {} does not work for image width {} and padding {}".format(filter_width, W, pad))
    if (H + 2 * pad - filter_height) % stride != 0:
        raise ValueError("Filter height {} does not work for image height {} and padding {}".format(filter_height, H, pad))
    out_height = (H + 2 * pad - filter_height) / stride + 1
    out_width = (W + 2 * pad - filter_width) / stride + 1

    imgs_cols = im2col_cython(imgs, w.shape[2], w.shape[3], pad, stride)
    # compute w * imgs_cols
    res = w.reshape((w.shape[0], -1)).dot(imgs_cols)
    res = res.reshape(w.shape[0], out_height, out_width, imgs.shape[0])
    res = np.transpose(res, axes = (3, 0, 1, 2))
    # return the output of the convolution
    # as well as the im2col of imgs
    # (as we need that in the backward pass)
    return res, imgs_cols


def bprop_conv(imgs,
               imgs_cols,
               convout_grad,
               filters,
               stride,
               pad):
    """ Back-propagate gradients of convolution
    imgs has shape (n_imgs, n_channels_in, img_h, img_w)
    imgs_cols is the pre-computed im2col of imgs
    filters has shape (n_channels_in, n_channels_out, img_h, img_w)
    convout has shape (n_imgs, n_channels_out, img_h, img_w)
    """
    N, C, H, W = imgs.shape
    w = np.transpose(filters, axes=(1,0,2,3))
    num_filters, _, filter_height, filter_width = w.shape
    # first reshape the output gradient such that the
    # first dimension is the number of filters and the
    # second dimension is the rest, this allows us to compute
    # the gradient with respect to the weights as a simple dot
    # product (as in a fully connected layer)
    out_grad_reshaped = convout_grad.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dW = out_grad_reshaped.dot(imgs_cols.T).reshape(w.shape)
    # flip dimensions to match the filter and divide by N for proper scaling
    dFilter = np.transpose(dW, axes=(1,0,2,3)) / N

    # next compute the gradient with respect to the input
    # which again is a simple dot product
    # as in a fully connected layer
    dImg_cols = w.reshape(num_filters, -1).T.dot(out_grad_reshaped)

    # we now simply have to call col2im on these in order
    # to distribute the gradients appropriately
    dImg = col2im_cython(dImg_cols, N, C, H, W, filter_height, filter_width, pad, stride)

    return dImg, dFilter
