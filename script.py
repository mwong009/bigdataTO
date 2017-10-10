import timeit
import theano
import numpy as np
import pandas as pd
import theano.tensor as T

from numpy.random import uniform
from six.moves import cPickle
from theano import shared
from optimizers import sgd, rmsprop, adadelta, nesterov_momentum

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

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
        activation=T.nnet.sigmoid):

        self.input = input
        self.activation = activation

        if W is None:
            mask = np.ones(shape=((n_in, n_out)), dtype=np.bool)
            # mask[:,-1] = 0
            W_mask = shared(mask.flatten(), name='W_mask', borrow=True)

            W_value = uniform(low=-np.sqrt(6/(n_in+n_out)),
                high=np.sqrt(6/(n_in+n_out)),
                size=(np.prod([n_in, n_out]),))
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

        else:
            self.output = self.activation(linalg)

class mlp(object):
    def __init__(self, input):
        self.input = input
        self.layers = []
        self.params = []
        self.masks = []

    def addLayer(self, n_in, n_out, activation):
        if len(self.layers) == 0:
            layer_input = self.input
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

class RBM(object):
    def __init__(self, input, n_visible, n_hidden, W=None, hbias=None, vbias=None):
        self.input = input
        self.theano_rng = T.shared_randomstreams.RandomStreams(2468)
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if W is None:
            mask = np.ones(shape=((n_visible, n_hidden)), dtype=np.bool)
            # mask[:,-1] = 0
            W_mask = shared(mask.flatten(), name='W_mask', borrow=True)

            W_value = uniform(low=-np.sqrt(6/(n_visible+n_hidden)),
                high=np.sqrt(6/(n_visible+n_hidden)),
                size=(np.prod([n_visible, n_hidden]),))
            W = shared(W_value * mask.flatten(), name='W', borrow=True)

        if hbias is None:
            mask = np.ones((n_hidden,), dtype=np.bool)
            # mask[-1] = 0
            b_mask = shared(mask, name='hbias_mask', borrow=True)

            b_value = np.zeros((n_hidden,), dtype=theano.config.floatX)
            b = shared(b_value * mask, name='hbias', borrow=True)

        if vbias is None:
            mask = np.ones((n_visible,), dtype=np.bool)
            # mask[-1] = 0
            b_mask = shared(mask, name='vbias_mask', borrow=True)

            b_value = np.zeros((n_visible,), dtype=theano.config.floatX)
            b = shared(b_value * mask, name='vbias', borrow=True)

        self.W = W
        self.W_matrix = self.W.reshape((n_visible, n_hidden))
        self.hbias = hbias
        self.vbias = vbias
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W_matrix) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return - hidden_term - vbias_term

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W_matrix) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
            n=1, p=h1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W_matrix.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
            n=1, p=v1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # for PCD, we initialize from the old state of the chain
		if persistent is None:
			chain_start = ph_sample
		else:
            chain_start = persistent

        # perform actual negative phase
		# in order to implement CD-k/PCD-k we need to scan over the
		# function that implements one gibbs step k times.
		# Read Theano tutorial on scan for more information :
		# http://deeplearning.net/software/theano/library/scan.html
		# the scan will return the entire Gibbs chain
		[pre_sigmoid_nvs, nv_means, nv_samples,
		 pre_sigmoid_nhs, nh_means, nh_samples], updates = \
			theano.scan(self.gibbs_hvh,
				# the None are place holders, saying that
				# chain_start is the initial state corresponding to the
				# 6th output
				outputs_info=[None,  None,  None, None, None, chain_start],
				n_steps=k)

        # not that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - \
               T.mean(self.free_energy(chain_end))

        grads = T.grad(cost, self.params, consider_constant=[chain_end])
        opt = sgd(model.params, masks=model.masks)
        updates = opt.updates(model.params, grads, 1e-3)

        if persistent is not None:
            # Note that this works only if persistent is a shared variable
			updates[persistent] = nh_samples[-1]
			# pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)

        else:
            monitoring_cost = self.reconstruction_cost(self.input, chain_end)

    def reconstruction_cost(self, v_sample, v_fantasy):
		""" reconstruction distance using L_2 (Euclidean distance) norm
		"""
		# L_2 norm
        return (v_sample - v_fantasy).norm(2)

    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(
            fe_xi_flip - fe_xi)))

        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

def shared_data(data):
    sharedData = shared(np.asarray(data, dtype=thenao.config.floatX),
        borrow=True, allow_downcast=True)
    return sharedData

def main():

    batch_size = 128
    n_hidden = 24
    n_epochs = 50

    print('loading data...')
    f = open('dataset.save', 'rb')
    loadedObj = cPickle.load(f)

    n_samples = loadedObj['mode'].shape[0]
    np.random.seed(2468)
    train_idx = np.random.permutation(n_samples)[:int(0.7*n_samples)]
    valid_idx = np.random.permutation(n_samples)[int(0.7*n_samples):]

    features = np.concatenate((loadedObj['scale_data'],
        loadedObj['binary_data'],
        loadedObj['occupation'],
        loadedObj['trip_purp'],
        loadedObj['region'].reshape(n_samples,-1),
        loadedObj['pd'].reshape(n_samples,-1)), axis=1)
    n_features = features.shape[1]

    targets = loadedObj['mode']
    n_targets = targets.max() + 1

    train_x = shared(features[train_idx])
    train_y = T.cast(shared(targets[train_idx]), 'int32')
    n_train_batches = train_idx.shape[0] // batch_size

    valid_x = shared(features[valid_idx])
    valid_y = T.cast(shared(targets[valid_idx]), 'int32')
    n_valid_batches = valid_idx.shape[0] // batch_size

    x = T.matrix('features')
    y = T.ivector('targets')
    idx = T.lscalar()

    model = mlp(x)
    model.addLayer(n_features, n_targets, T.nnet.softmax)
    #model.addLayer(n_hidden, 1, T.nnet.softmax)
    cost = model.negLogLikelihood(y)

    grads = T.grad(cost, model.params)
    opt = sgd(model.params, masks=model.masks)
    updates = opt.updates(model.params, grads, 1e-3)

    print('building symbolic functions...')
    train = theano.function([idx],
        outputs=model.errors(y),
        updates=updates,
        givens={
            x: train_x[idx*batch_size:(idx+1)*batch_size],
            y: train_y[idx*batch_size:(idx+1)*batch_size]},
        name='train function',
        allow_input_downcast=True,
        on_unused_input='ignore')

    valid = theano.function([idx],
        outputs=model.errors(y),
        givens={
            x: valid_x[idx*batch_size:(idx+1)*batch_size],
            y: valid_y[idx*batch_size:(idx+1)*batch_size]},
        name='validation function',
        allow_input_downcast=True,
        on_unused_input='ignore')

    print('... training the model')

    patience = 5 * n_train_batches
    patience_increase = 2
    threshold = 0.998
    validation_frequency = n_train_batches // 10
    best_error = np.inf
    best_model = None
    done_looping=False

    epoch = 0
    start_time = timeit.default_timer()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_idx in range(n_train_batches):
            minibatch_avg_cost = train(minibatch_idx)
            iter = (epoch - 1) * n_train_batches + minibatch_idx

            if (iter + 1) % validation_frequency == 0:
                error = [valid(i) for i in range(n_valid_batches)]
                this_error = np.mean(error)

                print('epoch %i %d/%d validation error %.4f p:%d' %
                    (epoch, (minibatch_idx + 1.), n_train_batches, this_error,
                    patience/n_train_batches))

                if this_error < best_error:
                    if this_error < (best_error * threshold):
                        patience = max(patience, iter * patience_increase)

                    best_error = this_error
                    best_model = model

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    with open('dataset.model', 'wb') as m:
        cPickle.dump(best_model, m, protocol=cPickle.HIGHEST_PROTOCOL)

    print('Optimization complete with best validation score of %.4f' %
        best_error)
    print('The code ran for %d epochs with %.3f epochs/sec' %
        (epoch, 1. * epoch / (end_time - start_time)))

if __name__ == '__main__':
    main()
