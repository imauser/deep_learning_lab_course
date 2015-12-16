import numpy as np

from ..layers import Layer, Parameterized, Activation, DTYPE

import conv_classic as cconv
import pool_classic as cpool
from .conv_op import forward_conv, bprop_conv
from .pool_op import forward_max_pool, bprop_max_pool

class Conv(Layer, Parameterized):
    """ 
    A convolutional layer that supports convolving an input tensor
    with a set of filters.
    We will, for now, assume that we always want zero padding around
    the input (which we indicate by saying we want a 'same' convolution).
    We assume that the input tensor is shaped:
    (batch_size, input_channels, height, width)
    That is for a 32x32 RGB image and batch size 64 we would have:
    (64, 3, 32, 32)
    
    Parameters
    ----------
    input_layer : a :class:`Layer` instance
    n_feats : the number of features or neurons for the conv layer
    filter_shape : a tuple specifying the filter shape, e.g. (5,5)
    strides : a tuple specifying the stride of the convolution, e.g. (1,1)
    init_stddev : a float specifying the standard deviation for weight init
    classic_conv: a switch to switch between the two cython convolution implementations
    padding_mode : either a string specifying 'same' for same convolutions 
                   or a number specifying the padding in both width and height
    activation_fun : a :class:`Activation` instance
    """
    def __init__(self, input_layer, n_feats,
                 filter_shape, init_stddev, strides=(1,1),
                 padding_mode='same',
                 activation_fun=Activation('relu'),
                 classic_conv=False):
        """
        Initialize convolutional layer.
        :parameters@param input_layer 
        
        """
        self.n_feats = n_feats
        self.filter_shape = filter_shape
        self.strides = strides
        self.init_stddev = init_stddev
        self.padding_mode = padding_mode
        self.input_layer = input_layer
        self.input_shape = input_layer.output_size()
        self.n_channels = self.input_shape[1]
        self.activation_fun = activation_fun
        self.classic_conv = classic_conv
        
        W_shape = (self.n_channels, self.n_feats) + self.filter_shape
        self.W = np.asarray(np.random.normal(size=W_shape, scale=self.init_stddev), dtype=DTYPE)
        self.b = np.zeros(self.n_feats, dtype=DTYPE)

    def fprop(self, input):
        # we cache the input
        self.last_input = input
        # This is were we actually do the convolution with W!
        # NOTE you have two options here
        # -> using the im2col version or the classic conv version
        #    I will use the former as it is slightly faster
        #    and leave the code for the latter commented out
        #    you can, if you want, switch between them using the
        #    classic_conv switch above
        if not self.classic_conv:
            if self.strides[0] != self.strides[1]:
                raise ValueError("Only square strides supported for im2col version")
            if self.padding_mode == 'same':
                pad = (self.filter_shape[0] - 1) // 2
            else:
                # otherwise assume padding_mode is a simple number
                pad = self.padding_mode
            convout, last_cols = forward_conv(input, self.W, self.strides[0], pad)
            # cache the im2col result
            self.last_cols = last_cols
        else:
            if self.padding_mode != 'same':
                raise ValueError("Only same convolutions supported for classic_conv")
            if self.strides[0] != 1 or self.strides[1] != 1:
                raise ValueError("Only stride 1 supported for classic conv")
            convout = np.empty(self.output_size(), dtype=DTYPE)
            cconv.forward_conv(input, self.W, convout)
        convout += self.b[np.newaxis, :, np.newaxis, np.newaxis]
        if self.activation_fun is not None:
            return self.activation_fun.fprop(convout)
        else:
            return convout

    def bprop(self, output_grad):
        if self.activation_fun == None:
            output_grad_pre = output_grad
        else:
            output_grad_pre = self.activation_fun.bprop(output_grad)
        last_input_shape = self.last_input.shape
        n_imgs = output_grad_pre.shape[0]
        self.db = np.sum(output_grad_pre, axis=(0, 2, 3)) / (n_imgs)
        if not self.classic_conv:
            if self.strides[0] != self.strides[1]:
                raise ValueError("Only square strides supported for im2col version")
            if self.padding_mode == 'same':
                if self.filter_shape[0] != self.filter_shape[1]:
                    raise ValueError("same convolutions only supported for square filters, " \
                                      + "otherwise specify padding by hand")
                pad = (self.filter_shape[0] - 1) // 2
            else:
                pad = self.padding_mode
            input_grad, self.dW =bprop_conv(self.last_input, self.last_cols, output_grad_pre, self.W, self.strides[0], pad)
        else:
            input_grad = np.empty(last_input_shape, dtype=DTYPE)
            self.dW = np.empty(self.W.shape, dtype=DTYPE)
            cconv.bprop_conv(self.last_input, output_grad_pre, self.W, input_grad, self.dW)
        return input_grad

    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db
    
    def output_size(self):
        self.input_shape = self.input_layer.output_size()
        if self.padding_mode == 'same':
            h = self.input_shape[2]
            w = self.input_shape[3]
        else:
            if self.classic_conv:
                raise NotImplementedError("Unknown padding mode {} for classic_conv".format(self.padding_mode))
            pad = self.padding_mode
            h = (self.input_shape[2] + 2 * pad - self.filter_shape[0]) / self.strides[0] + 1
            w = (self.input_shape[3] + 2 * pad - self.filter_shape[1]) / self.strides[1] + 1
        shape = (self.input_shape[0], self.n_feats, h, w)
        return shape


class Pool(Layer):
    """
    A pooling layer for dimensionality reduction.

    Parameters:
    -----------
    input_layer : a :class:`Layer` instance
    n_feats : the number of features or neurons for the conv layer
    pool_shape : a tuple specifying the pooling region size, e.g., (3,3)
    strides : a tuple specifying the stride of the pooling, e.g. (1,1),
              strides == pool_shape results in non-overlaping pooling
    mode : the pooling type (we only support max-pooling for now)
    classic_pool: for switching between the two pooling implementations
    """
    def __init__(self, input_layer, pool_shape=(2, 2), strides=(2, 2), mode='max', classic_pool=False):
        if mode != 'max':
            raise NotImplementedError("Only max-pooling currently implemented")
        self.mode = mode
        self.pool_h, self.pool_w = pool_shape
        self.stride_y, self.stride_x = strides
        self.input_layer = input_layer
        self.input_shape = input_layer.output_size()
        self.classic_pool = False

    def fprop(self, input):
        # we cache the input
        self.last_input = input

        if not self.classic_pool:
            if self.stride_y != self.stride_x:
                raise ValueError("Only square stride supported for im2col pool")
            poolout, self.last_cols, self.last_switches = forward_max_pool(input,
                                                                           self.pool_h,
                                                                           self.pool_w,
                                                                           self.stride_y)
        else:
            poolout = np.empty(self.output_size(), dtype=DTYPE)
            # cache the switches
            # which are the positions were the maximum was
            # we need those for doing the backwards pass!
            self.last_switches = np.empty(self.output_size()+(2,),
                                          dtype=np.int)
            cpool.forward_pool(input, poolout, self.last_switches,
                               self.pool_h, self.pool_w,
                               self.stride_y, self.stride_x)
        return poolout

    def bprop(self, output_grad):
        if not self.classic_pool:
            input_grad = bprop_max_pool(self.last_input, self.last_cols, self.last_switches,
                                        output_grad, self.pool_h, self.pool_w, self.stride_y)
        else:
            input_grad = np.zeros(self.last_input.shape, dtype=DTYPE)
            bprop_pool(output_grad, self.last_switches, input_grad)
        return input_grad
    
    def output_size(self):
        self.input_shape = self.input_layer.output_size()
        shape = (self.input_shape[0],
                 self.input_shape[1],
                 (self.input_shape[2] - self.pool_h) //self.stride_y + 1,
                 (self.input_shape[3] - self.pool_w) //self.stride_x + 1)
        return shape


class Flatten(Layer):
    """ 
    This is a simple layer that you can use to flatten
    the output from, for example, a convolution or pooling layer. 
    Such that you can put a fully connected layer on top!
    The result will always preserve the dimensionality along
    the zeroth axis (the batch size) and flatten all other dimensions!
    """

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_shape = input_layer.output_size()
        
    def fprop(self, input):
        self.last_input_shape = input.shape
        return np.reshape(input, (input.shape[0], -1))

    def bprop(self, output_grad):
        return np.reshape(output_grad, self.last_input_shape)

    def output_size(self):
        self.input_shape = self.input_layer.output_size()
        osize = (self.input_shape[0], np.prod(self.input_shape[1:]))
        return osize
