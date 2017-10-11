import theano.tensor as T
import numpy as np
import theano
from theano import shared


class nesterov_momentum(object):
    """ Stochastic Gradient Descent (SGD) with Nesterov momentum
    Generates update expressions of the form:
    velocity := momentum * velocity - learning_rate * gradient
    param := param - learning_rate * gradient
    """

    def __init__(self, params, masks=None, momentum=0.9):
        self.memory = []
        for param in params:
            value = param.get_value(borrow=True)
            accu = shared(np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
            self.memory.extend([accu])
        self.masks = masks
        self.momentum = momentum

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
        for n, (param, grad, mask) in enumerate(zip(params, grads, self.masks)):
            velocity = self.memory[n]

            sgd = - (learning_rate * grad)
            velocity_new = self.momentum * velocity + sgd
            updates.append((velocity, velocity_new))

            update = self.momentum * velocity_new + sgd

            select = np.arange(len(mask.eval()))[mask.eval()]

            # compute parameter update, using the 'old' delta_accu
            updates.append((param,
                T.inc_subtensor(param[select], update[select])))

        return updates
