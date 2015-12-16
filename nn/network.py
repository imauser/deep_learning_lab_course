import numpy as np
from .layers import Layer, Parameterized, Activation, one_hot, unhot, DTYPE

class NeuralNetwork:
    """ Our Neural Network container class.
    """
    def __init__(self, layers):
        self.layers = layers
        
    def _loss(self, X, Y):
        Y_pred = self.predict(X)
        return self.layers[-1].loss(Y, Y_pred)

    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        h_in = X
        for layer in self.layers:
            h_in = layer.fprop(h_in)
        Y_pred = h_in
        return Y_pred
    
    def backpropagate(self, Y, Y_pred, upto=0):
        """ Backpropagation of partial derivatives through 
            the complete network up to layer 'upto'
        """
        next_grad = self.layers[-1].input_grad(Y, Y_pred)
        for layer in reversed(self.layers[upto:-1]):
            next_grad = layer.bprop(next_grad)
        return next_grad
    
    def classification_error(self, X, Y, batch_size):
        """ Calculate error on the given data 
            assuming they are classes that should be predicted. 
        """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        n_err = 0
        for b in range(n_batches):
            batch_begin = b*batch_size
            batch_end = batch_begin+batch_size
            X_batch = X[batch_begin:batch_end]
            Y_batch = Y[batch_begin:batch_end]

            # Forward propagation
            Y_pred = unhot(self.predict(X_batch))
            n_err += np.sum(Y_pred != Y_batch)
        return float(n_err) / n_samples
    
    def sgd_epoch(self, X, Y, learning_rate, batch_size):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        # also track the loss during training
        loss = 0.
        for b in range(n_batches):
            batch_begin = b*batch_size
            batch_end = batch_begin+batch_size
            X_batch = X[batch_begin:batch_end]
            Y_batch = Y[batch_begin:batch_end]

            # Forward propagation
            Y_pred = self.predict(X_batch)

            # Back propagation
            self.backpropagate(Y_batch, Y_pred)

            loss += self.layers[-1].loss(Y_batch, Y_pred)
            
            # Update parameters
            for layer in self.layers:
                if isinstance(layer, Parameterized):
                    for param, grad in zip(layer.params(),
                                          layer.grad_params()):
                        param -= learning_rate*grad
        return loss / n_batches
    
    def gd_epoch(self, X, Y, learning_rate):
        return self.sgd_epoch(X, Y, learning_rate, X.size[0])
    
    def train(self, X, Y, Xvalid=None, Yvalid=None, 
              learning_rate=0.1, max_epochs=100, 
              batch_size=64, descent_type="sgd",
              y_one_hot=True, log_every=5):
        """ Train network on the given data. """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        if y_one_hot:
            Y_train = one_hot(Y)
        else:
            Y_train = Y
        print("... starting training")
        for e in range(max_epochs+1):
            if descent_type == "sgd":
                train_loss = self.sgd_epoch(X, Y_train, learning_rate, batch_size)
            elif descent_type == "gd":
                train_loss = self.gd_epoch(X, Y_train, learning_rate)
            else:
                raise NotImplemented("Unknown gradient descent type {}".format(descent_type))

            # Output some statistics
            if e % log_every == 0:
                train_error = self.classification_error(X, Y, batch_size)
                print('epoch {:4d}, loss {:.4f}, train error {:.4f}'.format(e, train_loss, train_error))
                if Xvalid is not None:
                    valid_error = self.classification_error(Xvalid, Yvalid, batch_size)
                    print('\t\t\t valid error {:.4f}'.format(valid_error))
            else:
                # only basic output that induces no additional computational costs
                print('epoch {:4d}, loss {:.4f}'.format(e, train_loss))
                    
    
    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        for l, layer in enumerate(self.layers):
            if isinstance(layer, Parameterized):
                print('checking gradient for layer {}'.format(l))
                for p, param in enumerate(layer.params()):
                    # we iterate through all parameters
                    param_shape = param.shape
                    # define functions for conveniently swapping
                    # out parameters of this specific layer and 
                    # computing loss and gradient with these 
                    # changed parametrs
                    def output_given_params(param_new):
                        """ A function that will compute the output 
                            of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # return computed loss
                        return self._loss(X, Y)

                    def grad_given_params(param_new):
                        """A function that will compute the gradient 
                           of the network given a set of parameters
                        """
                        # copy provided parameters
                        param[:] = np.reshape(param_new, param_shape)
                        # Forward propagation through the net
                        Y_pred = self.predict(X)
                        # Backpropagation of partial derivatives
                        self.backpropagate(Y, Y_pred, upto=l)
                        # return the computed gradient 
                        return np.ravel(self.layers[l].grad_params()[p])

                    # let the initial parameters be the ones that
                    # are currently placed in the network and flatten them
                    # to a vector for convenient comparisons, printing etc.
                    param_init = np.ravel(np.copy(param))
                    
                    # ####################################
                    #      compute the gradient with respect to
                    #      the initial parameters in two ways:
                    #      1) with grad_given_params()
                    #      2) with finite differences 
                    #         using output_given_params()
                    #         (as discussed in the lecture)
                    #      if your implementation is correct 
                    #      both results should be epsilon close
                    #      to each other!
                    # ####################################
                    epsilon = 1e-8
                    loss_base = output_given_params(param_init)
                    gparam_bprop = grad_given_params(param_init)
                    gparam_fd = np.zeros_like(param_init, dtype=DTYPE)
                    for i in range(len(param_init)):
                        param_init[i] += epsilon
                        gparam_fd[i] = (output_given_params(param_init) - loss_base) / (epsilon)
                        param_init[i] -= epsilon
                    
                    err = np.mean(np.abs(gparam_bprop - gparam_fd))
                    print('diff {:.2e}'.format(err))
                    assert(err < 10 * epsilon)
                    
                    # reset the parameters to their initial values
                    param[:] = np.reshape(param_init, param_shape)
