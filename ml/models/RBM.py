import theano
import pickle
import pandas as pd
import numpy as np
import theano.tensor as T
import theano.tensor.shared_randomstreams as rs

from collections import OrderedDict
from theano import shared
from numpy.random import seed, permutation, uniform
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

class RestrictedBoltzmannMachine(object):
    def __init__(self, optimizers=sgd):

        ''' model hyperparameters '''
        self.random_seed = 999
        self.batch_size = 100
        self.split = 0.7
        self.theano_rng = rs.RandomStreams(self.random_seed)
        self.optimizer = optimizers

        ''' Theano Tensor variables '''
        self.index = T.lscalar() # minibatch index tensor
        self.visibles = [] # list of tensor variables to visible units
        self.W_params = [] # list of weight params tensors
        self.vbias = []    # list of vbias params tensors
        self.hbias = []    # list of hbias params tensors
        self.masks = []    # list of tensor gradient masks
        self.params = []   # list of params for gradient calculation

        ''' Theano Shared variables '''
        self.train_visibles = []    # list of shared variables to inputs
        self.valid_visibles = []    # list of shared variables to inputs

        self.num_samples = 0        # total number of samples used
        self.num_hidden = 0         # number of hidden units
        self.shapes = []            # list of number of features
        self.types = []             # list of type for each feature

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
        else:
            raise NotImplementedError()

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

    def conditional_energy(self, visibles, validate_terms, valid_term):
        ''' P(y|x) = 1/Z * exp(b_k + sum_i(ln(1+exp(x_j*W_ij + c_i + W_ki))))
            vx_c = x_j*W_ij + c_i + W_ki
        '''
        vx_c = self.hbias[-1] # vx_c: (hid,)
        b_k = 0

        for i, (v, W, vbias, size, t) in enumerate(zip(visibles,
            self.W_params, self.vbias, self.shapes, self.types)):

            # get name of variable
            name = W.name

            # transform weight vector back into a matrix/tensor
            W = W.reshape(size + (self.num_hidden,))
            vbias = vbias.reshape(size)

            if name in validate_terms:
                if name == valid_term:
                    valid_feature = {'type': t, 'name': name, 'loc': i,
                        'target': v, 'W': W}
                    b_k += vbias
                    if t != 'scale':
                        vx_c += W  # (features, category, hidden)

            else:
                if t == 'category':
                    # for category features (n, 'x', hidden)
                    vx_c += T.tensordot(v, W, axes=[[1,2], [0,1]]
                        ).dimshuffle(0, 'x', 1)

                else:
                    # for scale and binary features (n, 'x', hidden)
                    vx_c += T.dot(v, W).dimshuffle(0, 'x', 1)

        # energy term to sum over hidden axis
        if valid_feature['type'] == 'scale':
            energy = b_k + T.dot(T.nnet.sigmoid(vx_c), valid_feature['W'].T)
        else:
            energy = b_k + T.sum(T.nnet.softplus(vx_c), axis=-1)
        return energy, valid_feature

    def errors(self, visibles, validate_terms):

        output_error = []
        for valid_term in validate_terms:

            energy, valid_feature = self.conditional_energy(visibles,
                validate_terms, valid_term)

            if valid_feature['type']  == 'category':
                # for categorical features (n, feature, category)
                probabilities = T.nnet.softmax(energy)
                y = T.argmax(valid_feature['target'], axis=-1).flatten()
                p = T.argmax(probabilities, axis=-1)
                error = T.mean(T.neq(y, p)) # accuracy

            elif valid_feature['type']  == 'scale':
                # for scale features (n, feature)
                norm = self.norms[valid_feature['name']]
                y = valid_feature['target'].flatten() * norm
                y_out = T.nnet.softplus(energy).flatten() * norm
                error = T.sqrt(T.mean(T.sqr(y_out - y))) # RMSE error

            elif valid_feature['type'] == 'binary':
                # for binary features (n, feature)
                y = valid_feature['target'].flatten()
                prob = T.nnet.sigmoid(energy)
                p = T.ceil(prob * 3) - 2.
                error = T.mean(T.neq(p, y)) # accuracy

            else:
                raise NotImplementedError()

            output_error.append(error)

        return output_error

    def prediction(self, visibles, validate_terms):

        output_prediction = []
        for valid_term in validate_terms:

            energy, valid_feature = self.conditional_energy(visibles,
                validate_terms, valid_term)

            if valid_feature['type']  == 'category':
                # for categorical features (n, feature, category)
                probabilities = T.nnet.softmax(energy)
                y = T.argmax(valid_feature['target'], axis=-1).flatten()
                p = T.argmax(probabilities, axis=-1)

                output_prediction.extend([y])
                output_prediction.extend([p])

            elif valid_feature['type']  == 'scale':
                # for scale features (n, feature)
                norm = self.norms[valid_feature['name']]
                y = valid_feature['target'].flatten() * norm
                y_out = T.nnet.softplus(energy).flatten() * norm

                output_prediction.extend([y])
                output_prediction.extend([y_out])

            elif valid_feature['type'] == 'binary':
                # for binary features (n, feature)
                y = valid_feature['target'].flatten()
                prob = T.nnet.sigmoid(energy)
                p = T.ceil(prob * 3) - 2.

                output_prediction.extend([y])
                output_prediction.extend([p])

            else:
                raise NotImplementedError()

        return output_prediction

    def free_energy(self, visibles):
        ''' free energy
            F(v) = - bv - sum(ln(1+exp(wx_b)))
            wx_b = Wv + h
        '''

        wx_b = self.hbias[-1]
        visible_term = 0

        for v, W, vbias, size in zip(visibles, self.W_params, self.vbias,
            self.shapes):

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
                visible_term += T.tensordot(v, vbias, axes=[[1,2], [0,1]])

        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)

        return - hidden_term - visible_term

    def sample_h_given_v(self, v0_samples):

        h1_preactivation = self.hbias[-1]

        # propagate upstream
        for v, W, size in zip(v0_samples, self.W_params, self.shapes):

            # transform weight vector back into a matrix/tensor
            W = W.reshape(size + (self.num_hidden,))

            if W.ndim == 2:
                # for weights with 2 dimensions (feature, hidden)
                h1_preactivation += T.dot(v, W)

            elif W.ndim == 3:
                # for weights with 3 dimensions (feature, category, hidden)
                h1_preactivation += T.tensordot(v, W, axes=[[1,2], [0,1]])

        # sample ~h given v
        h1_mean = T.nnet.sigmoid(h1_preactivation)
        h1_samples = self.theano_rng.binomial(size=h1_mean.shape, n=1,
           p=h1_mean, dtype=theano.config.floatX)

        return [h1_preactivation, h1_mean, h1_samples]

    def sample_v_given_h(self, h0_samples):

        v1_preactivation = []
        v1_mean = []
        v1_samples = []

        # propagate downstream
        for W, vbias, size, t in zip(self.W_params, self.vbias, self.shapes,
            self.types):

            # transform weight vector back into a matrix/tensor
            name = W.name
            W = W.reshape(size + (self.num_hidden,))
            vbias = vbias.reshape(size)

            if t == 'category':
                # for categorical features (feature, category, hidden)
                W = W.dimshuffle(0,2,1)
                preactivation = T.dot(h0_samples, W) + vbias
                v1_preactivation.append(preactivation)

                # softmax neural activation for 3-D tensors
                (d1, d2, d3) = preactivation.shape
                preactivation = preactivation.reshape((d1 * d2, d3))
                activation = T.nnet.softmax(preactivation)
                v1_mean.append(activation.reshape((d1, d2, d3)))

            else:
                W = W.T
                preactivation = T.dot(h0_samples, W) + vbias
                v1_preactivation.append(preactivation)

                if t == 'scale':
                    v1_mean.append(preactivation)

                elif t == 'binary':
                    v1_mean.append(T.nnet.sigmoid(preactivation))

                else:
                    raise NotImplementedError()

        # sample ~v given h
        for t, mean, pre in zip(self.types, v1_mean, v1_preactivation):

            if t == 'category':
                # for categorical features (n, feature, category)
                v1_sample = self.theano_rng.multinomial(pvals=mean,
                    dtype=theano.config.floatX)
                v1_samples.append(v1_sample)

            elif t == 'scale':
                # for scale features (n, feature)
                v1_sample = T.nnet.softplus(
                    self.theano_rng.normal(size=mean.shape, avg=pre,
                        std=T.nnet.sigmoid(mean), dtype=theano.config.floatX))
                v1_samples.append(v1_sample)

            elif t =='binary':
                # for binary features (n, feature)
                # flip features between [-1, 1]
                v1_sample = self.theano_rng.binomial(size=mean.shape,
                    p=mean, dtype=theano.config.floatX) * 2 - 1

                # flip features between [0, [-1, 1]]
                v1_sample = self.theano_rng.binomial(size=mean.shape,
                    p=T.abs_(mean*2-1), dtype=theano.config.floatX) * v1_sample

                v1_samples.append(v1_sample)

            else:
                raise NotImplementedError()

        return [v1_preactivation, v1_mean, v1_samples]

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

    def get_cost_updates(self, lr, persistent=None, k=1):

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
        nv_means = gibbs_chain[:num_tensors]
        nv_samples = gibbs_chain[num_tensors: 2*num_tensors]

        # extract only the end chain samples
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

        # calculate the gradients
        grads = T.grad(cost, self.params, consider_constant=chain_end_samples)
        opt = self.optimizer(self.params, masks=self.masks)
        updates = opt.updates(self.params, grads, lr)

        # update the Gibbs chain
        for update in updates:
            rbm_updates[update[0]] = update[1]

        # monitor the cross-entropy and L2 loss
        monitoring_cost = self.reconstruction_cost(chain_end_means,
            chain_end_samples)

        return monitoring_cost, rbm_updates

    def reconstruction_cost(self, chain_end_means, chain_end_samples):
        """ reconstruction distance using L_2 (Euclidean distance) norm
        """
        cost = 0
        for v, mean, sample, t in zip(self.visibles, chain_end_means,
            chain_end_samples, self.types):
            if t == 'scale':
                # L2 loss
                cost += T.mean((v - sample).norm(2))

            elif t == 'binary':
                cost += T.mean((v - sample).norm(2))

            elif t == 'category':
                # cross-entropy loss
                cost -= T.mean(T.sum(v*T.log(mean) + (1-v)*T.log(1-mean),
                    axis=-1))
            else:
                raise NotImplementedError()

        return cost

    def build_functions(self, lr=1e-3, k=10):

        print('building symbolic functions...')

        cost, updates = self.get_cost_updates(lr, None, k)

        # theano training functions
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

        self.valid_rbm = theano.function([],
            outputs=self.errors(self.valid_visibles, self.validate_terms),
            name='valid_rbm',
            allow_input_downcast=True,
            on_unused_input='ignore')

        self.predict_rbm = theano.function([],
            outputs=self.prediction(self.valid_visibles, self.validate_terms),
            name='predict_rbm',
            allow_input_downcast=True,
            on_unused_input='ignore')

    def load_variables(self, features, norms, n_hidden, validate):

        seed(self.random_seed)
        train_idx = None            # indexing arrays
        valid_idx = None            # indexing arrays

        self.norms = norms              # dict of {scale: norms} values
        self.validate_terms = validate  # list of terms to validate
        self.num_hidden = n_hidden      # int of number of hidden units

        print('loading variables...')

        for name, d in features.items():
            print(name, d['value'].shape, d['type'])

            feature = np.asarray(d['value'], dtype=theano.config.floatX)
            self.num_samples = max(self.num_samples, feature.shape[0])
            self.shapes.append(feature.shape[1:])
            self.types.append(d['type'])

            # training and validation random shuffling and splitting
            if train_idx is None:
                train_idx = permutation(self.num_samples)[
                    :int(self.split * self.num_samples)]
                self.num_train_batches = train_idx.shape[0] // self.batch_size

            if valid_idx is None:
                valid_idx = permutation(self.num_samples)[
                    int(self.split * self.num_samples):]
                self.num_valid_batches = valid_idx[0] // self.batch_size

            # load numpy arrays to list of shared variables
            self.train_visibles.append(shared(feature[train_idx]))
            self.valid_visibles.append(shared(feature[valid_idx]))

            # construct the neural network paths
            self.add_visible_unit(name, self.shapes[-1], self.num_hidden)

        print('validation terms:', self.validate_terms)

    def initialize_session(self, path):

        # initialize training parameters
        self.threshold = 0.998
        self.validation_freq = self.num_train_batches // 10
        self.best_errors = np.ones(len(self.validate_terms))*np.inf
        self.done_looping = False
        self.epoch = 0

        self.data_log = pd.DataFrame()
        self.data_log.index.name = 'iteration'
        self.data_output = pd.DataFrame()

        self.path = path + str(self.num_hidden)

    def one_train_step(self):

        self.epoch += 1
        this_cost = []

        # loop over minibatches
        for minibatch_index in range(self.num_train_batches):
            minibatch_avg_cost = self.train_rbm(minibatch_index)

            # get training cost
            this_cost += [minibatch_avg_cost]
            iter = (self.epoch - 1) * self.num_train_batches + minibatch_index

            # validate every n batches
            if (iter + 1) % self.validation_freq == 0:

                # log training statistics
                self.data_log.loc[iter, 'epoch'] = int(self.epoch)
                self.data_log.loc[iter, 'minibatch'] = int(minibatch_index)
                self.data_log.loc[iter, 'cost'] = np.round(
                    np.mean(this_cost), 5)

                print(self.data_log.iloc[-1:, :3])
                self.data_log.to_csv(self.path + 'training_stats.csv')

                # calculate validation error for p(y|x)
                errors = np.asarray(self.valid_rbm())

                # log training statistics
                for valid_term, error in zip(self.validate_terms, errors):
                    self.data_log.loc[iter, valid_term] = np.round(error, 3)
                    print(valid_term, 'validation error',
                        '{0:.3f}'.format(np.round(error, 3)))

                self.data_log.to_csv(self.path + 'training_stats.csv')

                # update benchmarks
                update_threshold = 0
                for i, (this_error, best_error) in enumerate(zip(errors,
                    self.best_errors * self.threshold)):
                    if this_error < best_error:
                        self.best_errors[i] = this_error
                        update_threshold += 1

                # save best model and predictions
                if update_threshold > 0:
                    best_model = [self.hbias, self.vbias, self.W_params,
                        self.params, self.masks]

                    with open(self.path + 'model.save', 'wb') as b:
                        pickle.dump(best_model, b,
                            protocol=pickle.HIGHEST_PROTOCOL)

                    predictions = self.predict_rbm()
                    predictions = np.asarray(predictions).reshape(
                        len(self.validate_terms) * 2, -1).T

                    for i, name in enumerate(self.validate_terms):
                        self.data_output[name] = predictions[:,i*2]
                        self.data_output[name+'_pred'] = predictions[:,i*2+1]

                    # save outputs
                    self.data_output.to_csv(self.path + 'predictions.csv')
