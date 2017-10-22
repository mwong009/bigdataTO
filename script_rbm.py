import os, sys, datetime
import timeit, pickle
import theano
import pandas as pd
import numpy as np
import theano.tensor as T

from collections import OrderedDict
from ml.models import RestrictedBoltzmannMachine
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

def main():
		
    now = datetime.datetime.now()
    path = sys.path[0] + '/results/' + str(now.date()) + '/'

    if not os.path.exists(os.path.dirname(path)):
	    os.makedirs(os.path.dirname(path))

    print('loading data...')
    f = open('dataset.save', 'rb')
    loadedObj, norms = pickle.load(f)

    rbm = RestrictedBoltzmannMachine(optimizers=sgd)
    rbm.batch_size = 20
    rbm.load_variables(loadedObj, norms, n_hidden=4,
        validate=['mode_prime', 'trip_purp', 'trip_km'])
    rbm.build_functions(lr=1e-3, k=1)

    print('training the model...')

    num_epochs = 100
    rbm.initialize_session(path)
    start_time = timeit.default_timer()
    while (rbm.epoch < num_epochs):
        rbm.one_train_step()

    end_time = timeit.default_timer()
    timedelta = end_time - start_time

    print('The code ran for %d epochs with %.3f epochs/sec' %
        (rbm.epoch, 1. * rbm.epoch / timedelta))

    print('Total training time: %d h %d m %.2f s' %
        (int(timedelta/3600), int(timedelta / 60) % 60, timedelta % 60))

if __name__ == '__main__':
    main()
