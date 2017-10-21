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

    rbm = RestrictedBoltzmannMachine(optimizers=sgd)
    rbm.batch_size = 20
    rbm.load_variables(loadedObj, norms, n_hidden=16,
        validate=['mode_prime', 'trip_purp', 'trip_km'])
    rbm.build_functions(lr=1e-3, k=1)

    print('training the model...')

    num_epochs = 100
    rbm.initialize_session()
    start_time = timeit.default_timer()
    while (rbm.epoch < num_epochs):
        rbm.one_train_step()

    end_time = timeit.default_timer()

    print('The code ran for %d epochs with %.3f epochs/sec' %
        (rbm.epoch, 1. * rbm.epoch / (end_time - start_time)))

if __name__ == '__main__':
    main()
