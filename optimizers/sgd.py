import theano.tensor as T
import numpy as np
import theano
from theano import shared


class sgd(object):
    """ Stochastic Gradient Descent (SGD)
    Generates update expressions of the form:
    param := param - learning_rate * gradient
    """

    def __init__(self, params, masks=None):
        self.masks = masks

    def updates(self, params, grads, learning_rate):
        """
        Parameters
        ----------
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        Returns
        -------
        list(tuple)
            A tuple mapping each parameter to its update expression
        """
        updates = []
        for param, grad, mask in zip(params, grads, self.masks):
            update = - (learning_rate * grad)
            select = np.arange(len(mask.eval()))[mask.eval()]

            # compute parameter update, using the 'old' delta_accu
            updates.append((param,
                T.inc_subtensor(param[select], update[select])))

        return updates
