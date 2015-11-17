import numpy as np

from ..layers import Layer, Parameterized

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
    init_stddef : a float specifying the standard deviation for weight init
    padding_mode : a string specifying which padding mode to use 
      (we only support zero padding or 'same' convolutions for now)
    """
    def __init__(self, input_layer, n_feats, init_stddev,
                 filter_shape, strides=(1,1),
                 padding_mode='same'):
        """
        Initialize convolutional layer.
        :parameters@param input_layer 
        
        """
        self.n_feats = n_feats
        self.filter_shape = filter_shape
        self.strides = strides
        self.init_stddev = init_stddev
        self.weight_decay = weight_decay
        self.padding_mode = padding_mode
        self.input_shape = input_layer.output_size()
        self.n_channels = self.input_shape[1]
        
        W_shape = (n_channels, self.n_feats) + self.filter_shape
        self.W = np.random.normal(size=W_shape, scale=self.init_stddev)
        self.b = np.zeros(self.n_feats)

    def fprop(self, input):
        # we cache the input and the input
        self.last_input = input
        convout = np.empty(self.output_shape(input.shape))
        # TODO
        # This is were you actually do the convolution with W!
        # You do not have to consider the bias!
        # We will simply add it in a second step (see line below)
        # you simply need to convolve the input with self.W and
        # write the result into convout
        # HINT: I recommend putting conv and pooling in little helper functions
        #       at the start of this file!
        #       The call to these should then look something like:
        #       conv(input, self.W, convout)
        # TODO
        return convout + self.b[np.newaxis, :, np.newaxis, np.newaxis]

    def bprop(self, output_grad):
        last_input_shape = self.last_input.shape
        input_grad = np.empty(last_input_shape)
        self.dW = np.empty(self.W.shape)
        # TODO
        # TODO:
        # This is were you have to backpropagate through the convolution
        # and write your results into dW
        # NOTE: again the bias is covered below!
        # HINT: I recommend putting conv and pooling in little helper functions
        #       at the start of this file!
        #       The call to these should then look something like:
        # bprop_conv(self.last_input, output_grad, self.W, input_grad,  self.dW)
        # TODO
        n_imgs = output_grad.shape[0]
        self.db = np.sum(output_grad, axis=(0, 2, 3)) / (n_imgs)
        return input_grad

    def params(self):
        return self.W, self.b

    def grad_params(self):
        return self.dW, self.db
    
    def output_size(self):
        if self.padding_mode == 'same':
            h = self.input_shape[2]
            w = self.input_shape[3]
        else:
            raise NotImplementedError("Unknown padding mode {}".format(self.padding_mode))
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
       stides == pool_shape results in non-overlaping pooling
    mode : the pooling type (we only support max-pooling for now)
    """
    def __init__(self, input_layer, pool_shape=(3, 3), strides=(1, 1), mode='max'):
        if mode != 'max':
            raise NotImplementedError("Only max-pooling currently implemented")
        self.mode = mode
        self.pool_h, self.pool_w = pool_shape
        self.stride_y, self.stride_x = strides
        self.input_shape = input_layer.output_size()

    def fprop(self, input):
        # we cache the input
        self.last_input_shape = input.shape
        # and also the switches
        # which are the positions were the maximum was
        # we need those for doing the backwards pass!
        self.last_switches = np.empty(self.output_shape(input.shape)+(2,),
                                      dtype=np.int)
        poolout = np.empty(self.output_shape(input.shape))
        # TODO
        # this is were you have to implement pooling
        # HINT: it is very similar to the convolution from above
        #       only that you compute a max rather than a multiplication with
        #       weights
        # HINT: You should store the result in poolout and the max positions
        #       (switches) in self.last_switches, you will need those in the
        #       backward pass!
        # the call should look something like:
        # pool(input, poolout, self.last_switches, self.pool_h, self.pool_w,
        #      self.stride_y, self.stride_x)
        # TODO
        return poolout

    def bprop(self, output_grad):
        input_grad = np.empty(self.last_input_shape)
        # TODO
        # implement the backward pass through the pooling
        # it should use the switches, the call should look something like:
        # bprop_pool(output_grad, self.last_switches, input_grad)
        # TODO
        return input_grad
    
    def output_size(self):
        input_shape = self.input_shape
        shape = (input_shape[0],
                 input_shape[1],
                 input_shape[2]//self.stride_y,
                 input_shape[3]//self.stride_x)
        return shape


class Flatten(Layer):
    """ 
    This is a simple layer that you can use to flatten
    the output from, for example, a convolution or pooling layer. 
    Such that you can put a fully connected layer on top!
    The result will always preserve the dimensionality along
    the zeroth axis (the batch size) and flatten all other dimensions!
    """
    def fprop(self, input):
        self.last_input_shape = input.shape
        return np.reshape(input, (input.shape[0], -1))

    def bprop(self, output_grad):
        return np.reshape(output_grad, self.last_input_shape)

    def output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))
