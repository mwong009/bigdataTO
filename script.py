import timeit
import pickle
import theano
import numpy as np
import theano.tensor as T

from ml.models import MultiLayerPerceptron
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

# class RBM(object):
#     def __init__(self, input, n_visible, n_hidden, W=None, hbias=None, vbias=None):
#         self.input = input
#         self.theano_rng = T.shared_randomstreams.RandomStreams(2468)
#         self.n_visible = n_visible
#         self.n_hidden = n_hidden
#
#         if W is None:
#             mask = np.ones(shape=((n_visible, n_hidden)), dtype=np.bool)
#             # mask[:,-1] = 0
#             W_mask = shared(mask.flatten(), name='W_mask', borrow=True)
#
#             W_value = uniform(low=-np.sqrt(6/(n_visible+n_hidden)),
#                 high=np.sqrt(6/(n_visible+n_hidden)),
#                 size=(np.prod([n_visible, n_hidden]),))
#             W = shared(W_value * mask.flatten(), name='W', borrow=True)
#
#         if hbias is None:
#             mask = np.ones((n_hidden,), dtype=np.bool)
#             # mask[-1] = 0
#             b_mask = shared(mask, name='hbias_mask', borrow=True)
#
#             b_value = np.zeros((n_hidden,), dtype=theano.config.floatX)
#             b = shared(b_value * mask, name='hbias', borrow=True)
#
#         if vbias is None:
#             mask = np.ones((n_visible,), dtype=np.bool)
#             # mask[-1] = 0
#             b_mask = shared(mask, name='vbias_mask', borrow=True)
#
#             b_value = np.zeros((n_visible,), dtype=theano.config.floatX)
#             b = shared(b_value * mask, name='vbias', borrow=True)
#
#         self.W = W
#         self.W_matrix = self.W.reshape((n_visible, n_hidden))
#         self.hbias = hbias
#         self.vbias = vbias
#         self.params = [self.W, self.hbias, self.vbias]
#
#     def free_energy(self, v_sample):
#         wx_b = T.dot(v_sample, self.W_matrix) + self.hbias
#         vbias_term = T.dot(v_sample, self.vbias)
#         hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
#         return - hidden_term - vbias_term
#
#     def propup(self, vis):
#         pre_sigmoid_activation = T.dot(vis, self.W_matrix) + self.hbias
#         return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
#
#     def sample_h_given_v(self, v0_sample):
#         pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
#         # get a sample of the hiddens given their activation
#         # Note that theano_rng.binomial returns a symbolic sample of dtype
#         # int64 by default. If we want to keep our computations in floatX
#         # for the GPU we need to specify to return the dtype floatX
#         h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
#             n=1, p=h1_mean, dtype=theano.config.floatX)
#         return [pre_sigmoid_h1, h1_mean, h1_sample]
#
#     def propdown(self, hid):
#         pre_sigmoid_activation = T.dot(hid, self.W_matrix.T) + self.vbias
#         return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
#
#     def sample_v_given_h(self, h0_sample):
#         pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
#         # get a sample of the visible given their activation
#         # Note that theano_rng.binomial returns a symbolic sample of dtype
#         # int64 by default. If we want to keep our computations in floatX
#         # for the GPU we need to specify to return the dtype floatX
#         v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
#             n=1, p=v1_mean, dtype=theano.config.floatX)
#         return [pre_sigmoid_v1, v1_mean, v1_sample]
#
#     def gibbs_hvh(self, h0_sample):
#         ''' This function implements one step of Gibbs sampling,
#             starting from the hidden state'''
#         pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
#         pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
#         return [pre_sigmoid_v1, v1_mean, v1_sample,
#                 pre_sigmoid_h1, h1_mean, h1_sample]
#
#     def gibbs_vhv(self, v0_sample):
#         ''' This function implements one step of Gibbs sampling,
#             starting from the visible state'''
#         pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
#         pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
#         return [pre_sigmoid_h1, h1_mean, h1_sample,
#                 pre_sigmoid_v1, v1_mean, v1_sample]
#
#     def get_cost_updates(self, lr=0.1, persistent=None, k=1):
#         pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
#
#         # for PCD, we initialize from the old state of the chain
#         if persistent is None:
#             chain_start = ph_sample
#         else:
#             chain_start = persistent
#
#         # perform actual negative phase
# 		# in order to implement CD-k/PCD-k we need to scan over the
# 		# function that implements one gibbs step k times.
# 		# Read Theano tutorial on scan for more information :
# 		# http://deeplearning.net/software/theano/library/scan.html
# 		# the scan will return the entire Gibbs chain
#         [pre_sigmoid_nvs, nv_means, nv_samples,
#         pre_sigmoid_nhs, nh_means, nh_samples], updates =\
#             theano.scan(self.gibbs_hvh,
# 				# the None are place holders, saying that
# 				# chain_start is the initial state corresponding to the
# 				# 6th output
#                 outputs_info=[None,  None,  None, None, None, chain_start],
#                 n_steps=k)
#
#         # not that we only need the sample at the end of the chain
#         chain_end = nv_samples[-1]
#
#         cost = T.mean(self.free_energy(self.input)) - \
#                T.mean(self.free_energy(chain_end))
#
#         grads = T.grad(cost, self.params, consider_constant=[chain_end])
#         opt = sgd(model.params, masks=model.masks)
#         updates = opt.updates(model.params, grads, 1e-3)
#
#         if persistent is not None:
#             # Note that this works only if persistent is a shared variable
#             updates[persistent] = nh_samples[-1]
# 			# pseudo-likelihood is a better proxy for PCD
#             monitoring_cost = self.get_pseudo_likelihood_cost(updates)
#
#         else:
#             monitoring_cost = self.reconstruction_cost(self.input, chain_end)
#
#     def reconstruction_cost(self, v_sample, v_fantasy):
#         """ reconstruction distance using L_2 (Euclidean distance) norm
#         """
# 		# L_2 norm
#         return (v_sample - v_fantasy).norm(2)
#
#     def get_pseudo_likelihood_cost(self, updates):
#         bit_i_idx = theano.shared(value=0, name='bit_i_idx')
#         xi = T.round(self.input)
#         fe_xi = self.free_energy(xi)
#         xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
#         fe_xi_flip = self.free_energy(xi_flip)
#
#         # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
#         cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(
#             fe_xi_flip - fe_xi)))
#
#         updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
#         return cost
#

def main():
    print('loading data...')
    f = open('dataset.save', 'rb')
    loadedObj = pickle.load(f)

    features = np.concatenate((loadedObj['scale_data'],
        loadedObj['binary_data'],
        loadedObj['occupation'],
        loadedObj['trip_purp'],
        loadedObj['region'].reshape(loadedObj['region'].shape[0],-1),
        loadedObj['pd'].reshape(loadedObj['pd'].shape[0],-1)), axis=1)
    targets = loadedObj['mode']
    hiddens = [100]

    mlp = MultiLayerPerceptron()
    mlp._set_hyperparameters('batch_size', 128)
    mlp._set_hyperparameters('learning_rate', 1e-3)
    mlp._set_hyperparameters('optimizer', sgd)

    mlp.load_variables(features, targets)

    mlp.add_layer(mlp.num_features, hiddens[0], T.nnet.sigmoid)
    mlp.add_layer(hiddens[0], mlp.num_targets, T.nnet.softmax)

    mlp.build_functions()

    print('training the model...')

    num_epochs = 50
    mlp.initialize_session()
    start_time = timeit.default_timer()
    while (mlp.epoch < num_epochs) and (not mlp.done_looping):
        mlp.one_train_step()

    end_time = timeit.default_timer()

    with open('dataset.model', 'wb') as m:
        pickle.dump(mlp.best_model, m, protocol=pickle.HIGHEST_PROTOCOL)

    print('Optimization complete with best validation score of %.4f' %
        best_error)
    print('The code ran for %d epochs with %.3f epochs/sec' %
        (epoch, 1. * epoch / (end_time - start_time)))

if __name__ == '__main__':
    main()
