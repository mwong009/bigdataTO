import timeit
import pickle
import theano
import numpy as np
import theano.tensor as T

from collections import OrderedDict
from ml.models import RestrictedBoltzmannMachine
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

def main():
    print('loading data...')
    f = open('dataset_rbm.save', 'rb')
    loadedObj = pickle.load(f)

    features = OrderedDict()

    features['scale_data'] = loadedObj['scale_data']

    features['binary_data'] = loadedObj['binary_data']

    features['mode'] = np.eye(loadedObj['mode'].max()+1)[loadedObj['mode']]
    features['mode'] = features['mode'].reshape((-1, 1,
        features['mode'].shape[1]))

    features['occupation'] = np.eye(loadedObj['occupation'].max()+1)[
        loadedObj['occupation']]
    features['occupation'] = features['occupation'].reshape((-1, 1,
        features['occupation'].shape[1]))

    features['trip_purp'] = np.eye(loadedObj['trip_purp'].max()+1)[
        loadedObj['trip_purp']]
    features['trip_purp'] = features['trip_purp'].reshape((-1, 1,
        features['trip_purp'].shape[1]))

    features['region'] = np.eye(loadedObj['region'].max()+1)[
        loadedObj['region']]

    features['pd'] = np.eye(loadedObj['pd'].max()+1)[loadedObj['pd']]

    rbm = RestrictedBoltzmannMachine()
    rbm.load_variables(features, n_hidden=120)
    rbm.build_train_functions(lr=1e-3, k=2)
    #rbm.build_valid_functions(chain_steps=1)

    print('training the model...')

    num_epochs = 1
    rbm.initialize_session()
    start_time = timeit.default_timer()
    while (rbm.epoch < num_epochs):
        rbm.one_train_step()

    end_time = timeit.default_timer()

    # with open('dataset.model', 'wb') as m:
    #     pickle.dump(mlp.best_model, m, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # print('Optimization complete with best validation score of %.4f' %
    #     best_error)
    print('The code ran for %d epochs with %.3f epochs/sec' %
        (rbm.epoch, 1. * rbm.epoch / (end_time - start_time)))

    # target_samples = np.argmax(rbm.valid_y[0].eval(), axis=-1)
    # output = rbm.valid_rbm()
    # pred_samples = np.argmax(output[2], axis = -1)
    #
    # print(np.equal(pred_samples.flatten(), target_samples.flatten()).mean())
    #
    # with open('model.output', 'wb') as m:
    #     pickle.dump([target_samples.flatten(), pred_samples.flatten()], m,
    #         protocol=pickle.HIGHEST_PROTOCOL)
    #
    # print(output)

if __name__ == '__main__':
    main()
