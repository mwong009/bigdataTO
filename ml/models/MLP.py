import theano
import numpy as np
import theano.tensor as T

from theano import shared
from numpy.random import seed, permutation, uniform
from ml.models.HiddenLayer import HiddenLayer
from ml.models.LogisticSoftmax import LogisticSoftmax
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

class MultiLayerPerceptron(object):
    def __init__(self, optimizers=sgd, seed=9999):
        self.batch_size = 100
        self.split = 0.7
        self.random_seed = 999
        self.learning_rate = 1e-3
        self.layers = []
        self.params = []
        self.masks = []

        self.x = T.matrix('features')
        self.y = T.ivector('targets')
        self.index = T.lscalar()

        self.optimizer = optimizers

    def _set_hyperparameters(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            raise KeyError()

    def add_layer(self, n_in, n_out, activation):
        if len(self.layers) == 0:
            layer_input = self.x
        else:
            layer_input = self.layers[-1].output

        if activation is T.nnet.softmax:
            layer = LogisticSoftmax(layer_input, n_in, n_out, W=None, b=None,
                activation=T.nnet.softmax)
        else:
            layer = HiddenLayer(layer_input, n_in, n_out, W=None, b=None,
                activation=activation)

        self.layers.append(layer)
        self.params.extend(layer.params)
        self.masks.extend(layer.masks)

    def negLogLikelihood(self, target):
        prob = self.layers[-1].output
        return -T.mean(T.log(prob)[T.arange(target.shape[0]), target])

    def errors(self, target):
        pred = self.layers[-1].pred
        if target.ndim != pred.ndim:
            raise TypeError('targets should have the same shape as output',
                ('target', target.type, 'pred', pred.type))

        if target.dtype.startswith('int'):
            return T.mean(T.neq(pred, target))
        else:
            raise NotImplementedError()

    def load_variables(self, features, targets):

        print('loading variables...')

        self.features = features
        self.targets = targets

        self.num_samples = features.shape[0]
        self.num_features = features.shape[1]
        self.num_targets = targets.max() + 1

        seed(self.random_seed)
        train_idx = permutation(self.num_samples)[
            :int(self.split * self.num_samples)]
        valid_idx = permutation(self.num_samples)[
            int(self.split * self.num_samples):]

        self.train_x = shared(features[train_idx])
        self.train_y = T.cast(shared(targets[train_idx]), 'int32')
        self.num_train_batches = train_idx.shape[0] // self.batch_size

        self.valid_x = shared(features[valid_idx])
        self.valid_y = T.cast(shared(targets[valid_idx]), 'int32')
        self.num_valid_batches = valid_idx.shape[0] // self.batch_size

    def build_functions(self):

        print('building symbolic functions...')

        cost = self.negLogLikelihood(self.y)

        grads = T.grad(cost, self.params)
        opt = self.optimizer(self.params, masks=self.masks)
        updates = opt.updates(self.params, grads, self.learning_rate)

        self.train_model = theano.function([self.index],
            outputs=self.errors(self.y),
            updates=updates,
            givens={
                self.x: self.train_x[self.index * self.batch_size:
                    (self.index + 1) * self.batch_size],
                self.y: self.train_y[self.index * self.batch_size:
                    (self.index + 1) * self.batch_size]
            },
            name='training_function',
            allow_input_downcast=True,
            on_unused_input='ignore')

        self.validate_model = theano.function([self.index],
            outputs=self.errors(self.y),
            givens={
                self.x: self.valid_x[self.index * self.batch_size:
                    (self.index + 1) * self.batch_size],
                self.y: self.valid_y[self.index * self.batch_size:
                    (self.index + 1) * self.batch_size]
            },
            name='validation_function',
            allow_input_downcast=True,
            on_unused_input='ignore')

    def initialize_session(self):
        self.patience = 5 * self.num_train_batches
        self.patience_increase = 2
        self.threshold = 0.998
        self.validation_freq = self.num_train_batches // 10
        self.best_model = None
        self.best_error = np.inf
        self.done_looping = False
        self.epoch = 0
        self.epoch_score = []

    def one_train_step(self):
        self.epoch += 1
        for minibatch_index in range(self.num_train_batches):
            minibatch_avg_cost = self.train_model(minibatch_index)
            iter = (self.epoch - 1) * self.num_train_batches + minibatch_index

            if (iter + 1) % self.validation_freq == 0:
                this_error = np.mean(
                    [self.validate_model(i) for i in range(
                        self.num_valid_batches)])

                print('epoch %i %d/%d validation error %.4f patience:%d' %
                    (self.epoch, (minibatch_index + 1), self.num_train_batches,
                    this_error, self.patience/self.num_train_batches))

                if this_error < self.best_error:
                    if this_error < (self.best_error * self.threshold):
                        self.patience = max(self.patience,
                            iter * self.patience_increase)

                    self.best_error = this_error
                    self.best_model = self.layers
                    self.epoch_score.append([self.epoch, iter, this_error])

                if self.patience <= iter:
                    self.done_looping = True
                    break
