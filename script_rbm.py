import timeit
import pickle
import theano
import pandas as pd
import numpy as np
import theano.tensor as T

from collections import OrderedDict
from ml.models import RestrictedBoltzmannMachine
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

def main():
    print('loading data...')
    f = open('dataset.save', 'rb')
    loadedObj, norms = pickle.load(f)

    features = OrderedDict()

    for name, var in loadedObj.items():
        features[name] = var

    rbm = RestrictedBoltzmannMachine()
    rbm.batch_size = 200
    rbm.load_variables(features, norms, n_hidden=20, validate=['mode_prime'])
    rbm.build_functions(lr=1e-2, k=2)

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
