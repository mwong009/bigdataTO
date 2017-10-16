import timeit
import pickle
import theano
import numpy as np
import theano.tensor as T

from ml.models import MultiLayerPerceptron
from ml.optimizers import sgd, rmsprop, adadelta, nesterov_momentum

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
