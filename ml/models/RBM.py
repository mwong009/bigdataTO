import theano
import numpy as np
import theano.tensor as T

from theano import shared
from numpy.random import seed, permutation, uniform
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

class RestrictedBoltzmannMachine(object):
    def __init__(self, optimizers=sgd, n_hidden=5):
        self.random_seed = 999
        self.batch_size = 1000
        self.split = 0.7
        self.theano_rng = T.shared_randomstreams.RandomStreams(2468)

        self.index = T.lscalar() # minibatch index tensor

        self.visibles = [] # list of tensor variables to visible units
        self.W_params = [] # list of weight params tensors
        self.vbias = []    # list of vbias params tensors
        self.hbias = []    # list of hbias params tensors
        self.masks = []    # list of tensor gradient masks
        self.params = []   # list of params for gradient calculation

        self.n_hidden = n_hidden
        self.optimizer = optimizers

    def add_softmax_visible_unit(self, name, n_visibles, n_hidden):

        n_visible, n_category = n_visibles

        if len(self.hbias) == 0:
            mask = np.ones((n_hidden,), dtype=np.bool)
            hbias_mask = shared(mask, name='hbias_mask', borrow=True)

            hbias_value = np.zeros((n_hidden,), dtype=theano.config.floatX)
            hbias = shared(hbias_value * mask, name='hbias', borrow=True)

            self.hbias.append(hbias)
            self.params.extend([hbias])
            self.masks.extend([hbias_mask])

        mask = np.ones(shape=((n_visible, n_category, n_hidden)),
            dtype=np.bool)
        W_mask = shared(mask.flatten(), name='W_' + name + '_mask',
            borrow=True)

        W_value = uniform(low=-np.sqrt(6/(n_visible+n_category+n_hidden)),
            high=np.sqrt(6/(n_visible+n_category+n_hidden)),
            size=(np.prod([n_visible, n_category, n_hidden]),))
        W = shared(W_value, name='W_' + name, borrow=True)

        mask = np.ones((n_visible, n_category), dtype=np.bool)
        vbias_mask = shared(mask.flatten(), name='vbias_' + name + 'mask',
            borrow=True)

        vbias_value = np.zeros(np.prod([n_visible, n_category]),
            dtype=theano.config.floatX)
        vbias = shared(vbias_value * mask.flatten(), name='vbias_' + name, borrow=True)

        self.visibles.append(T.tensor3('visibles'))
        self.W_params.append(W)
        self.vbias.append(vbias)
        self.params.extend([W, vbias])
        self.masks.extend([W_mask, vbias_mask])

    def add_visible_unit(self, name, n_visible, n_hidden):

        if type(n_visible) is tuple:
            n_visible = n_visible[0]

        if len(self.hbias) == 0:
            mask = np.ones((n_hidden,), dtype=np.bool)
            hbias_mask = shared(mask, name='hbias_mask', borrow=True)

            hbias_value = np.zeros((n_hidden,), dtype=theano.config.floatX)
            hbias = shared(hbias_value * mask, name='hbias', borrow=True)

            self.hbias.append(hbias)
            self.params.extend([hbias])
            self.masks.extend([hbias_mask])

        mask = np.ones(shape=((n_visible, n_hidden)), dtype=np.bool)
        W_mask = shared(mask.flatten(), name='W_' + name + '_mask', borrow=True)

        W_value = uniform(low=-np.sqrt(6/(n_visible+n_hidden)),
            high=np.sqrt(6/(n_visible+n_hidden)),
            size=(np.prod([n_visible, n_hidden]),))
        W = shared(W_value, name='W_' + name, borrow=True)

        mask = np.ones((n_visible,), dtype=np.bool)
        vbias_mask = shared(mask, name='vbias_' + name + 'mask', borrow=True)

        vbias_value = np.zeros((n_visible,), dtype=theano.config.floatX)
        vbias = shared(vbias_value * mask, name='vbias_' + name, borrow=True)

        self.visibles.append(T.matrix('visibles'))
        self.W_params.append(W)
        self.vbias.append(vbias)
        self.params.extend([W, vbias])
        self.masks.extend([W_mask, vbias_mask])

    def free_energy(self, visibles):
        wx_b = self.hbias[-1]
        vbias_term = 0
        for v_sample, W_param, vbias_param, sh in zip(visibles, self.W_params, self.vbias, self.num_features):
            W_matrix = W_param.reshape(sh+(self.n_hidden,))
            if v_sample.ndim == 2:
                wx_b += T.dot(v_sample, W_matrix)
                vbias_term += T.dot(v_sample, vbias_param)
            elif v_sample.ndim == 3:
                wx_b += T.tensordot(v_sample, W_matrix, axes=[[1,2],[0,1]])
                vbias_term += T.tensordot(v_sample, vbias_param.reshape(sh),
                    axes=[[1,2],[0,1]])

        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)

        return - hidden_term - vbias_term

    def propup(self, visibles):
        pre_activation = self.hbias[-1]
        for vis, W_param, sh in zip(visibles, self.W_params, self.num_features):
            W_matrix = W_param.reshape(sh+(self.n_hidden,))
            if W_matrix.ndim == 2:
                pre_activation += T.dot(vis, W_matrix)
            elif W_matrix.ndim == 3:
                pre_activation += T.tensordot(vis, W_matrix,
                    axes=[[1,2],[0,1]])
        return [pre_activation, T.nnet.sigmoid(pre_activation)]

    def sample_h_given_v(self, v0_samples):
        pre_sigmoid_h1, h1_mean = self.propup(v0_samples)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_samples = self.theano_rng.binomial(size=h1_mean.shape,
            n=1, p=h1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_samples]

    def propdown(self, hidden):
        pre_activation = []
        post = []
        for W_param, vbias, sh in zip(self.W_params,
            self.vbias, self.num_features):
            W_matrix = W_param.reshape(sh+(self.n_hidden,))
            if W_matrix.ndim == 2:
                activation = T.dot(hidden, W_matrix.T) + vbias
                pre_activation.append(activation)
                post.append(T.nnet.sigmoid(activation))
            elif W_matrix.ndim == 3:
                activation = T.dot(hidden,
                    W_matrix.dimshuffle(0,2,1)) + vbias.reshape(sh)
                (d1, d2, d3) = activation.shape
                pre_activation.append(activation)
                post.append(T.nnet.softmax(activation.reshape(
                    (d1*d2, d3))).reshape((d1,d2,d3)))
        return [pre_activation, post]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_samples = []
        for pre_sigmoid, mean in zip(pre_sigmoid_v1, v1_mean):
            if pre_sigmoid.ndim == 2:
                v1_samples.append(self.theano_rng.binomial(
                    size=mean.shape, n=1, p=mean,
                    dtype=theano.config.floatX))
            elif pre_sigmoid.ndim == 3:
                v1_samples.append(self.theano_rng.multinomial(pvals=mean,
                    dtype=theano.config.floatX))
        return [pre_sigmoid_v1, v1_mean, v1_samples]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_samples = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_samples)

        return pre_sigmoid_v1 + v1_mean + v1_samples + \
            [pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_samples):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_samples)
        pre_sigmoid_v1, v1_mean, v1_samples = self.sample_v_given_h(h1_sample)

        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean,
            v1_samples]

    def get_new_samples(self, validation_samples):

        [pre_sigmoid_nhs, nh_means, nh_samples,
         pre_sigmoid_nvs, nv_means, nv_samples] = self.gibbs_vhv(
            validation_samples)

        return nv_samples, nv_means

    def get_cost_updates(self, lr=1e-3, persistent=None, k=20):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(
            self.visibles)

        # number of tensors to iterate over
        num_varray = len(self.visibles)

        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # [pre_sigmoid_nvs] + [nv_means] + [nv_samples] +
        # [pre_sigmoid_nhs, nh_means, nh_samples]
        outputs, rbm_updates =\
            theano.scan(self.gibbs_hvh,
				# the None are place holders, saying that
				# chain_start is the initial state corresponding to the
				# [-1] output
                outputs_info=3*[None]*num_varray + [None, None, chain_start],
                n_steps=k,
                name='gibbs_hvh')

        # not that we only need the sample at the end of the chain
        nv_samples = outputs[-(3+num_varray):-3]
        nv_means = outputs[num_varray:2*num_varray]
        chain_end = []
        chain_end_means = []
        for nv, means in zip(nv_samples, nv_means):
            chain_end.append(nv[-1])
            chain_end_means.append(means[-1])

        model_cost = T.mean(self.free_energy(chain_end))
        data_cost = T.mean(self.free_energy(self.visibles))
        cost = data_cost - model_cost

        grads = T.grad(cost, self.params, consider_constant=chain_end)
        opt = self.optimizer(self.params, masks=self.masks)
        updates = opt.updates(self.params, grads, lr)
        for update in updates:
            rbm_updates[update[0]] = update[1]

        monitoring_cost = self.reconstruction_cost(chain_end_means)
        return monitoring_cost, rbm_updates

    def reconstruction_cost(self, v_pred_m):
        """ reconstruction distance using L_2 (Euclidean distance) norm
        """
		# L_2 norm
        cost = 0
        for sample, pred_m in zip(self.visibles, v_pred_m):
            if sample.ndim == 2:
                cost += T.mean((sample - pred_m).norm(2))
            elif sample.ndim == 3:
                cost -= T.mean(T.sum(sample * T.log(pred_m) +
                    (1-sample) * T.log(1-pred_m), axis=-1))
                #cost += T.mean(T.neq(sample, pred))
        return cost

    def build_train_functions(self, lr=1e-3, k=5):

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

        chain_end, mf = self.get_new_samples(self.valid_x)

        pred = T.argmax(chain_end[2], axis=-1)
        targ = T.argmax(self.valid_y, axis=-1)

        self.valid_rbm = theano.function([],
            outputs=T.mean(T.neq(pred, targ)),
            name='valid_rbm',
            allow_input_downcast=True,
            on_unused_input='ignore')

    def load_variables(self, features, n_hidden):

        seed(self.random_seed)
        self.train_visibles = []    # list of shared variables to inputs
        self.valid_x = []           # list of shared variables to inputs
        self.valid_y = []
        self.num_features = []      # list of number of features in each input
        self.num_samples = 0        # total number of samples used
        self.n_hidden = n_hidden
        train_idx = None            # indexing array
        valid_idx = None

        print('loading variables...')
        for name, feature in features.items():
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

            if name == 'mode':
                self.valid_y.append(shared(feature[valid_idx]))
                feature = np.zeros(feature.shape, dtype=theano.config.floatX)
            self.valid_x.append(shared(feature[valid_idx]))

            if len(feature.shape[1:]) == 1:
                self.add_visible_unit(name,
                    self.num_features[-1], self.n_hidden)
            elif len(feature.shape[1:]) == 2:
                self.add_softmax_visible_unit(name,
                    self.num_features[-1], self.n_hidden)

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

    def one_train_step(self):
        self.epoch += 1

        this_cost = []
        for minibatch_index in range(self.num_train_batches):
            minibatch_avg_cost = self.train_rbm(minibatch_index)
            this_cost += [minibatch_avg_cost]
            iter = (self.epoch - 1) * self.num_train_batches + minibatch_index

            if (iter + 1) % self.validation_freq == 0:
                print('epoch %i %d/%d cost is %.2f' %
                    (self.epoch, (minibatch_index + 1), self.num_train_batches,
                    np.mean(this_cost)))

                this_error = self.valid_rbm()

                print('  validation error %.4f' % (this_error))
