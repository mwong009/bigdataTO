import theano
import numpy as np
import theano.tensor as T

from theano import shared
from numpy.random import uniform

class LogisticSoftmax(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
        activation=T.nnet.softmax):

        self.input = input
        self.activation = activation

        if W is None:
            mask = np.ones(shape=((n_in, n_out)), dtype=np.bool)
            # mask[:,-1] = 0
            W_mask = shared(mask.flatten(), name='W_mask', borrow=True)

            W_value = uniform(low=-1., high=1., size=(np.prod([n_in, n_out]),))
            W = shared(W_value * mask.flatten(), name='W', borrow=True)

        if b is None:
            mask = np.ones((n_out,), dtype=np.bool)
            # mask[-1] = 0
            b_mask = shared(mask, name='b_mask', borrow=True)

            b_value = np.zeros((n_out,), dtype=theano.config.floatX)
            b = shared(b_value * mask, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

        self.W_mask = W_mask
        self.b_mask = b_mask
        self.masks = [self.W_mask, self.b_mask]

        W_matrix = self.W.reshape((n_in, n_out))
        linalg = T.dot(self.input, W_matrix) + self.b

        if self.activation is None:
            self.output = linalg
            self.pred = self.output

        else:
            self.output = self.activation(linalg)
            self.pred = T.argmax(self.output, axis=1)
