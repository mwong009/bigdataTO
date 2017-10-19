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
        self.batch_size = 200
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

    def conditional_energy(self, visibles, validate_terms, valid_term):
        '''
            P(y|x) = 1/Z * exp(b_k + sum_i(ln(1+exp(x_j*W_ij + c_i + W_ki))))
            vx_c = x_j*W_ij + c_i + W_ki
        '''
        valid_feature = {}
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
                    if t == 'scale':
                        valid_feature['type'] = t
                        valid_feature['name'] = name
                        valid_feature['loc'] = i
                        target = v
                        v = shared(np.zeros(v.shape.eval(),
                                dtype=theano.config.floatX),
                            name=name, borrow=True)
                        visibles[i] = v

                    else:
                        valid_feature['type'] = t
                        valid_feature['name'] = name
                        vx_c += W
                        b_k += vbias
                        target = v

            else:
                if t == 'category':
                    # for category features (n, 'x', hidden)
                    vx_c += T.tensordot(v, W, axes=[[1,2], [0,1]]
                        ).dimshuffle(0, 'x', 1)

                else:
                    # for scale and binary features (n, 'x', hidden)
                    vx_c += T.dot(v, W).dimshuffle(0, 'x', 1)

        # energy term to sum over hidden axis
        energy = b_k + T.sum(T.nnet.softplus(vx_c), axis=-1)

        return energy, valid_feature, target, visibles

    def errors(self, visibles, validate_terms):

        output_error = []
        for valid_term in validate_terms:

            e, valid_feature, target, visibles = self.conditional_energy(
                visibles, validate_terms, valid_term)

            if valid_feature['type']  == 'category':
                # for categorical features (n, feature, category)
                probabilities = T.nnet.softmax(e)
                y = T.argmax(target, axis=-1).flatten()
                p = T.argmax(probabilities, axis=-1).flatten()
                error = T.mean(T.neq(y, p)) # accuracy

                print(valid_feature['name'], p.eval(), y.eval())

            elif valid_feature['type']  == 'scale':
                # for scale features (n, feature)
                y = target.flatten() * self.norms[valid_feature['name']]
                # y_out = T.nnet.relu(energy).flatten()
                gibbs_output = self.gibbs_vhv(visibles)
                y_out = T.nnet.relu(gibbs_output[2:2+len(visibles)][
                    valid_feature['loc']]).flatten() * self.norms[
                    valid_feature['name']]
                error = T.sqrt(T.mean(T.sqr(y_out - y))) # RMSE error

                print(valid_feature['name'], y_out.eval(), y.eval())

            elif valid_feature['type'] == 'binary':
                # for binary features (n, feature)
                y = target.flatten()
                prob = T.nnet.sigmoid(e)
                p = T.ceil(prob * 3) - 2.
                error = T.mean(T.neq(p, y)) # accuracy

                print(valid_feature['name'], p.eval(), y.eval())

            else:
                raise NotImplementedError()

            output_error.append(error)

        return output_error

    def prediction(self, visibles, validate_terms):

        output_prediction = []
        for valid_term in validate_terms:

            e, valid_feature, target, visibles = self.conditional_energy(
                visibles, validate_terms, valid_term)

            if valid_feature['type']  == 'category':
                # for categorical features (n, feature, category)
                probabilities = T.nnet.softmax(e)
                y = T.argmax(target, axis=-1).flatten()
                p = T.argmax(probabilities, axis=-1).flatten()

                output_prediction.extend([y])
                output_prediction.extend([p])

            elif valid_feature['type']  == 'scale':
                # for scale features (n, feature)
                y = target.flatten() * self.norms[valid_feature['name']]
                # y_out = T.nnet.relu(energy).flatten()
                gibbs_output = self.gibbs_vhv(visibles)
                y_out = T.nnet.relu(gibbs_output[2:2+len(visibles)][
                    valid_feature['loc']]).flatten() * self.norms[
                    valid_feature['name']]

                output_prediction.extend([y])
                output_prediction.extend([y_out])

            elif valid_feature['type'] == 'binary':
                # for binary features (n, feature)
                y = target.flatten()
                prob = T.nnet.sigmoid(e)
                p = T.ceil(prob * 3) - 2.
                error = T.mean(T.neq(p, y)) # accuracy

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
                visible_term += T.tensordot(v, vbias, axes=[[1,2],[0,1]])

        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return - hidden_term - visible_term

    def sample_h_given_v(self, v0_samples):

        pre_sigmoid_h1 = self.hbias[-1]

        for v, W, size in zip(v0_samples, self.W_params, self.shapes):

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
        # h1_samples = T.nnet.relu(self.theano_rng.normal(size=h1_mean.shape,
        #    avg=pre_sigmoid_h1, std=1.0, dtype=theano.config.floatX))

        return [pre_sigmoid_h1, h1_mean, h1_samples]

    def sample_v_given_h(self, h0_samples):

        v1_preactivation = []
        v1_mean = []
        v1_samples = []

        for W, vbias, size, t in zip(self.W_params, self.vbias, self.shapes, self.types):

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

            elif t == 'scale':
                W = W.T
                preactivation = T.dot(h0_samples, W) + vbias
                v1_preactivation.append(preactivation)
                v1_mean.append(preactivation)

            elif t =='binary':
                W = W.T
                preactivation = T.dot(h0_samples, W) + vbias
                v1_preactivation.append(preactivation)
                v1_mean.append(T.nnet.sigmoid(preactivation))

            else:
                raise NotImplementedError()

        for t, mean, pre in zip(self.types, v1_mean, v1_preactivation):

            if t == 'category':
                # for categorical features (n, feature, category)
                v1_sample = self.theano_rng.multinomial(pvals=mean,
                    dtype=theano.config.floatX)
                v1_samples.append(v1_sample)

            elif t == 'scale':
                # for scale features (n, feature)
                v1_sample = T.nnet.relu(
                    self.theano_rng.normal(size=mean.shape, avg=pre,
                        std=T.nnet.sigmoid(mean),
                        dtype=theano.config.floatX))
                # v1_sample = mean
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

        grads = T.grad(cost, self.params, consider_constant=chain_end_samples)
        opt = self.optimizer(self.params, masks=self.masks)
        updates = opt.updates(self.params, grads, lr)
        for update in updates:
            rbm_updates[update[0]] = update[1]

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
                cost += T.mean((v - mean).norm(2))
            elif t == 'binary':
                cost += T.mean((v - sample).norm(2))
            elif t == 'category':
                cost -= T.mean(T.sum(v * T.log(mean) +
                    (1-v) * T.log(1-mean), axis=-1))
            else:
                raise NotImplementedError()
        return cost

    def build_functions(self, lr=1e-3, k=5):

        print('building symbolic functions...')

        cost, updates = self.get_cost_updates(lr, None, k)

        # theano training function
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
        self.train_visibles = []    # list of shared variables to inputs
        self.valid_visibles = []    # list of shared variables to inputs
        self.shapes = []            # list of number of features
        self.types = []             # list of type for each feature

        self.num_samples = 0        # total number of samples used
        self.num_hidden = n_hidden
        train_idx = None            # indexing array
        valid_idx = None

        self.norms = norms
        self.validate_terms = validate

        print('loading variables...')

        for name, d in features.items():
            print(name, d['value'].shape, d['type'])

            feature = np.asarray(d['value'], dtype=theano.config.floatX)
            self.num_samples = max(self.num_samples, feature.shape[0])
            self.shapes.append(feature.shape[1:])
            self.types.append(d['type'])

            if train_idx is None:
                train_idx = permutation(self.num_samples)[
                    :int(self.split * self.num_samples)]
            if valid_idx is None:
                valid_idx = permutation(self.num_samples)[
                    int(self.split * self.num_samples):]

            self.train_visibles.append(shared(feature[train_idx]))
            self.valid_visibles.append(shared(feature[valid_idx]))

            self.add_visible_unit(name, self.shapes[-1], self.num_hidden)

        self.num_train_batches = train_idx.shape[0] // self.batch_size
        self.num_valid_batches = valid_idx[0] // self.batch_size

        print('validate terms:', self.validate_terms)

    def initialize_session(self):
        self.patience = 5 * self.num_train_batches
        self.patience_increase = 2
        self.threshold = 0.998
        self.validation_freq = self.num_train_batches // 10
        self.best_error = np.asarray([np.inf])
        self.done_looping = False
        self.epoch = 0
        self.epoch_score = []
        self.data_log = pd.DataFrame()
        self.data_log.index.name = 'iteration'
        self.data_output = pd.DataFrame()

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
                errors = self.valid_rbm()

                for valid_term, error in zip(self.validate_terms, errors):
                    self.data_log.loc[iter, valid_term] = np.round(error, 3)
                    print(valid_term, 'validation error',
                        '{0:.3f}'.format(np.round(error, 3)))

                self.data_log.to_csv('training_stats.csv')

                errors = np.asarray(errors)

                if (errors < (self.threshold * self.best_error)).any():
                    self.best_error = errors
                    best_model = [self.hbias, self.vbias, self.W_params,
                        self.params, self.masks]

                    with open('model.save', 'wb') as b:
                        pickle.dump(best_model, b,
                            protocol=pickle.HIGHEST_PROTOCOL)

                    preditions = self.predict_rbm()
                    preditions = np.asarray(preditions).reshape(-1,
                        len(self.validate_terms*2))

                    for i, name in enumerate(zip(self.validate_terms)):

                        self.data_output[name] = preditions[:,i*2]
                        self.data_output[name+'_pred'] = preditions[:,i*2+1]

                    self.data_output.to_csv(path_or_buf='predictions.csv')

        df = pd.DataFrame(self.output_samples)
        df.to_csv('predictions.csv')
        # with open('model.output', 'wb') as m:
        #     pickle.dump(samples, m, protocol=pickle.HIGHEST_PROTOCOL)
