import theano
import pickle
import pandas as pd
import numpy as np
import theano.tensor as T

from theano import shared
from numpy.random import seed, permutation, uniform
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

class RestrictedBoltzmannMachine(object):
    def __init__(self, optimizers=sgd):

        ''' model hyperparameters '''
        self.random_seed = 999
        self.batch_size = 1000
        self.split = 0.7
        self.theano_rng = T.shared_randomstreams.RandomStreams(2468)
        self.optimizer = optimizers

        ''' Theano Tensor variables '''
        self.index = T.lscalar() # minibatch index tensor
        self.visibles = [] # list of tensor variables to visible units
        self.W_params = [] # list of weight params tensors
        self.vbias = []    # list of vbias params tensors
        self.hbias = []    # list of hbias params tensors
        self.masks = []    # list of tensor gradient masks
        self.params = []   # list of params for gradient calculation

    def add_hbias(self, n_hidden):

        mask = np.ones((n_hidden,), dtype=np.bool)
        hbias_mask = shared(mask, borrow=True)

        hbias_value = np.zeros((n_hidden,), dtype=theano.config.floatX)
        hbias = shared(hbias_value * mask, name='hbias', borrow=True)

        # update model parameters
        self.hbias.append(hbias)
        self.params.extend([hbias])
        self.masks.extend([hbias_mask])

    def add_visible_unit(self, name, n_visible, n_hidden):
        ''' parameters
            :tuple n_visible: length of visible units
            :int n_hidden: length of hidden units
        '''

        # size of weight tensor
        size = n_visible + (n_hidden,)

        if len(n_visible) == 1:
            tensor_variable = T.matrix(name)
        elif len(n_visible) == 2:
            tensor_variable = T.tensor3(name)

        if len(self.hbias) == 0:
            self.add_hbias(n_hidden)

        # create masks for W params
        mask = np.ones(size, dtype=np.bool)

        W_mask = shared(mask.flatten(), borrow=True)

        # create W params with masking
        W_value = uniform(low=-np.sqrt(6/np.sum(size)),
            high=np.sqrt(6/np.sum(size)),
            size=(np.prod(size),))

        W = shared(W_value * mask.flatten(), name=name, borrow=True)

        # create masks for vbias params
        mask = np.ones(n_visible, dtype=np.bool)

        vbias_mask = shared(mask.flatten(), borrow=True)

        # create vbias params
        vbias_value = np.zeros(np.prod(n_visible), dtype=theano.config.floatX)

        vbias = shared(vbias_value * mask.flatten(), name=name, borrow=True)

        # update model parameters
        self.visibles.append(tensor_variable)
        self.W_params.append(W)
        self.vbias.append(vbias)
        self.params.extend([W, vbias])
        self.masks.extend([W_mask, vbias_mask])

    def discriminative_cost(self, visibles, samples):
        '''
            P(y|x) = 1/Z * exp(v_k + sum_i(ln(1+exp(W_ki + h_k + x_j W_ij))))
        '''
        vx_c = self.hbias[0]
        visible_term = 0

        for v, W, vbias, size in zip(samples, self.W_params, self.vbias,
            self.num_features):

            # transform weight vector back into a matrix/tensor
            W = W.reshape(size + (self.num_hidden,))
            if W.name == 'mode_prime':
                # W (feature, category, hidden)
                vx_c += W
                visible_term += vbias

            else:
                if W.ndim == 2:
                    # (batch, 'x', hidden)
                    vx_c += T.dot(v, W).dimshuffle(0, 'x', 1)

                elif W.ndim == 3:
                    # (batch, 'x', hidden)
                    vx_c += T.tensordot(v, W, axes=[[1,2], [0,1]]).dimshuffle(
                        0, 'x', 1)

        hidden_term = T.sum(T.log(1 + T.exp(vx_c)), axis=-1)
        energy = visible_term + hidden_term

        prob = T.nnet.softmax(energy)

        for v, W in zip(visibles, self.W_params):
            if W.name == 'mode_prime':
                # negative log likelihood
                y = T.argmax(v, axis=-1)
                cost = -T.mean(T.log(prob)[T.arange(y.shape[0]), y])

        return cost

    def pred(self, visibles, name):
        '''
            P(y|x) = 1/Z * exp(b_k + sum_i(ln(1+exp(x_j*W_ij + c_i + W_ki))))
            vx_c = x_j*W_ij + c_i + W_ki
        '''
        vx_c = self.hbias[-1]
        energy = 0

        for v, W, vbias, size in zip(visibles, self.W_params, self.vbias,
            self.num_features):

            W_name = W.name
            # transform weight vector back into a matrix/tensor
            W = W.reshape(size + (self.num_hidden,))
            vbias = vbias.reshape(size)

            if W.ndim == 2:
                # for weights with 2 dimensions (feature, hidden)
                xW = T.dot(v, W).dimshuffle(0, 'x', 1)

            elif W.ndim == 3:
                # for weights with 3 dimensions (feature, category, hidden)
                xW = T.tensordot(v, W, axes=[[1,2],[0,1]]).dimshuffle(0, 'x', 1)

            if W_name == name:
                vx_c += W
                energy += vbias

            else:
                vx_c += xW

        energy += T.sum(T.log(1+T.exp(vx_c)), axis=-1)
        prob = T.nnet.softmax(energy)

        return [prob]


    def free_energy(self, visibles):
        ''' free energy
            F(v) = - bv - sum(ln(1+exp(wx_b)))
            wx_b = Wv + h
        '''

        wx_b = self.hbias[-1]
        visible_term = 0

        for v, W, vbias, size in zip(visibles, self.W_params, self.vbias,
            self.num_features):

            # transform weight vector back into a matrix/tensor
            W = W.reshape(size + (self.num_hidden,))
            vbias = vbias.reshape(size)

            if W.ndim == 2:
                # for weights with 2 dimensions (feature, hidden)
                wx_b += T.dot(v, W)
                visible_term += T.dot(v, vbias)

            elif W.ndim == 3:
                # for weights with 3 dimensions (feature, category, hidden)
                wx_b += T.tensordot(v, W, axes=[[1,2],[0,1]])
                visible_term += T.tensordot(v, vbias, axes=[[1,2],[0,1]])

        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return - hidden_term - visible_term

    def sample_h_given_v(self, v0_samples):

        pre_sigmoid_h1 = self.hbias[-1]

        for v, W, size in zip(v0_samples, self.W_params, self.num_features):

            # transform weight vector back into a matrix/tensor
            W = W.reshape(size + (self.num_hidden,))

            if W.ndim == 2:
                # for weights with 2 dimensions (feature, hidden)
                pre_sigmoid_h1 += T.dot(v, W)

            elif W.ndim == 3:
                # for weights with 3 dimensions (feature, category, hidden)
                pre_sigmoid_h1 += T.tensordot(v, W, axes=[[1,2],[0,1]])

        h1_mean = T.nnet.sigmoid(pre_sigmoid_h1)
        h1_samples = self.theano_rng.binomial(size=h1_mean.shape, n=1,
            p=h1_mean, dtype=theano.config.floatX)

        return [pre_sigmoid_h1, h1_mean, h1_samples]

    def sample_v_given_h(self, h0_samples):

        pre_activation_v1 = []
        v1_mean = []
        v1_samples = []

        for W, vbias, size in zip(self.W_params, self.vbias, self.num_features):

            # transform weight vector back into a matrix/tensor
            W = W.reshape(size + (self.num_hidden,))
            vbias = vbias.reshape(size)

            if W.ndim == 2:
                # for weights with 2 dimensions (feature, hidden)
                pre_sigmoid_v1 = T.dot(h0_samples, W.T) + vbias
                pre_activation_v1.append(pre_sigmoid_v1)

                # sigmoid neural activation for 2-D matrix
                v1_mean.append(T.nnet.sigmoid(pre_sigmoid_v1))

            elif W.ndim == 3:
                # for weights with 3 dimensions (feature, category, hidden)
                pre_softmax_v1 = T.dot(h0_samples, W.dimshuffle(0,2,1))
                pre_softmax_v1 += vbias
                pre_activation_v1.append(pre_softmax_v1)

                # softmax neural activation for 3-D tensors
                (d1, d2, d3) = pre_softmax_v1.shape
                v1_mean.append(T.nnet.softmax(pre_softmax_v1.reshape(
                    (d1 * d2, d3))).reshape((d1, d2, d3)))

        for v, mean in zip(self.visibles, v1_mean):

            if mean.ndim == 2:
                # for tensors with 2 dimensions (batch, feature)
                if v.name == 'binary_variables':
                    # scale binary samples to [-1, 1]
                    v1_samples.append(T.ceil(3 * mean) - 2.)
                else:
                    # scale variables use mean
                    v1_samples.append(mean)

            elif mean.ndim == 3:
                # for tensors with 3 dimensions (batch, feature, category)
                v1_samples.append(self.theano_rng.multinomial(pvals=mean,
                    dtype=theano.config.floatX))

        return [pre_activation_v1, v1_mean, v1_samples]

    def gibbs_hvh(self, h0_samples):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        _, v1_means, v1_samples = self.sample_v_given_h(h0_samples)
        _, h1_means, h1_samples = self.sample_h_given_v(v1_samples)

        return v1_means + v1_samples + [h1_means, h1_samples]

    def gibbs_vhv(self, v0_samples):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        _, h1_means, h1_samples = self.sample_h_given_v(v0_samples)
        _, v1_means, v1_samples = self.sample_v_given_h(h1_samples)

        return [h1_means, h1_samples] + v1_means + v1_samples

    def get_v1_samples(self, v0_samples):

        output = self.gibbs_vhv(v0_samples)
        v1_means = output[2: len(v0_samples)]
        v1_samples = output[(2+len(v0_samples)):]

        return v1_means, v1_samples

    def get_cost_updates(self, lr=1e-3, persistent=None, k=20):

        # perform positive phase
        _, ph_mean, ph_sample = self.sample_h_given_v(self.visibles)

        # number of tensors to iterate over
        num_tensors = len(self.visibles)

        # start of Gibbs sampling chain
        chain_start = ph_sample

        # perform negative phase
        gibbs_chain, rbm_updates = theano.scan(self.gibbs_hvh,
            outputs_info=2 * [None] * num_tensors + [None, chain_start],
            n_steps=k, name='gibbs_hvh')

        # not that we only need the sample at the end of the chain
        nv_samples = gibbs_chain[num_tensors: 2*num_tensors]
        nv_means = gibbs_chain[:num_tensors]

        chain_end_samples = []
        chain_end_means = []
        for samples, means in zip(nv_samples, nv_means):
            chain_end_samples.append(samples[-1])
            chain_end_means.append(means[-1])

        # calculate the model cost
        ''' cost = F_data(v) - F_model(v)'''
        model_cost = T.mean(self.free_energy(chain_end_samples))
        data_cost = T.mean(self.free_energy(self.visibles))
        cost = data_cost - model_cost

        # discriminative RBM cost
        #disc_cost = self.discriminative_cost(self.visibles, chain_end_samples)

        grads = T.grad(cost, self.params, consider_constant=chain_end_samples)
        opt = self.optimizer(self.params, masks=self.masks)
        updates = opt.updates(self.params, grads, lr)
        for update in updates:
            rbm_updates[update[0]] = update[1]

        monitoring_cost = self.reconstruction_cost(chain_end_means)
        return monitoring_cost, rbm_updates

    def reconstruction_cost(self, chain_end_means):
        """ reconstruction distance using L_2 (Euclidean distance) norm
        """
        cost = 0
        for samples, means in zip(self.visibles, chain_end_means):
            if samples.ndim == 2:
                # L2 loss
                cost += T.mean((samples - means).norm(2))
            elif samples.ndim == 3:
                # cross-entropy loss
                cost -= T.mean(T.sum(samples * T.log(means) +
                    (1-samples) * T.log(1-means), axis=-1))

        return cost

    def build_functions(self, lr=1e-3, k=5):

        print('building symbolic functions...')

        cost, updates = self.get_cost_updates(lr=lr,
            persistent=None, k=k)

        self.train_rbm = theano.function([self.index],
            outputs=cost,
            updates=updates,
            givens={
                v: train[self.index * self.batch_size:
                    (self.index + 1) * self.batch_size] for v, train in \
                    zip(self.visibles, self.train_visibles)
                },
            name='train_rbm',
            allow_input_downcast=True,
            on_unused_input='ignore')

        v1_means, v1_samples = self.get_v1_samples(self.valid_x)

        accuracies = []
        # for i, valid_y in zip(self.valid_y_idx, self.valid_y):
        #     target = T.argmax(valid_y, axis=-1)
        #     pred = T.argmax(v1_samples[i], axis=-1)
        #     accuracies.append(T.mean(T.neq(pred, target)))

        probs = self.pred(self.valid_x, 'mode_prime')
        for prob, valid_y in zip(probs, self.valid_y):
            target = T.argmax(valid_y, axis=-1).flatten()
            pred = T.argmax(prob, axis=-1)
            accuracies.append(T.mean(T.neq(pred, target)))

        self.valid_rbm = theano.function([],
            outputs=accuracies,
            name='valid_rbm',
            allow_input_downcast=True,
            on_unused_input='ignore')

        self.output_samples = theano.function([],
            outputs=probs,
            name='output_samples',
            allow_input_downcast=True,
            on_unused_input='ignore')

    def load_variables(self, features, n_hidden):

        seed(self.random_seed)
        self.train_visibles = []    # list of shared variables to inputs
        self.valid_x = []           # list of shared variables to inputs
        self.valid_y = []
        self.valid_y_idx = []
        self.valid_y_names = []
        self.num_features = []      # list of number of features in each input
        self.num_samples = 0        # total number of samples used
        self.num_hidden = n_hidden
        train_idx = None            # indexing array
        valid_idx = None

        print('loading variables...')
        for i, (name, feature) in enumerate(features.items()):
            print(i, name, feature.shape)
            feature = np.asarray(feature, dtype=theano.config.floatX)
            self.num_samples = max(self.num_samples, feature.shape[0])
            self.num_features.append(feature.shape[1:])

            if train_idx is None:
                train_idx = permutation(self.num_samples)[
                    :int(self.split * self.num_samples)]
            if valid_idx is None:
                valid_idx = permutation(self.num_samples)[
                    int(self.split * self.num_samples):]

            self.train_visibles.append(shared(feature[train_idx]))

            self.add_visible_unit(name, self.num_features[-1], self.num_hidden)

            if name == 'mode_prime':
                self.valid_y_idx.append(i)
                self.valid_y_names.append(name)
                self.valid_y.append(shared(feature[valid_idx]))
                feature = np.zeros(feature.shape, dtype=theano.config.floatX)
            self.valid_x.append(shared(feature[valid_idx]))

        self.num_train_batches = train_idx.shape[0] // self.batch_size
        self.num_valid_batches = valid_idx[0] // self.batch_size

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
        self.data_log = pd.DataFrame()
        self.data_log.index.name = 'iteration'

    def one_train_step(self):
        self.epoch += 1
        this_cost = []
        for minibatch_index in range(self.num_train_batches):
            minibatch_avg_cost = self.train_rbm(minibatch_index)
            this_cost += [minibatch_avg_cost]
            iter = (self.epoch - 1) * self.num_train_batches + minibatch_index

            if (iter + 1) % self.validation_freq == 0:
                self.data_log.loc[iter, 'epoch'] = int(self.epoch)
                self.data_log.loc[iter, 'minibatch'] = int(minibatch_index)
                self.data_log.loc[iter, 'cost'] = np.round(
                    np.mean(this_cost), 5)

                print(self.data_log.iloc[-1:, :3])

                self.data_log.to_csv('training_stats.csv')

            if (iter + 1) % (self.validation_freq * 1) == 0:
                this_error = self.valid_rbm()
                for name, error in zip(self.valid_y_names, this_error):
                    self.data_log.loc[iter, name] = np.round(error, 5)
                    print(name, 'validation error',
                        '{0:.3f}%'.format(100 * np.round(error, 5)))
                self.data_log.to_csv('training_stats.csv')

        output = self.output_samples()
        with open('model.output', 'wb') as m:
            pickle.dump(samples, m, protocol=pickle.HIGHEST_PROTOCOL)
