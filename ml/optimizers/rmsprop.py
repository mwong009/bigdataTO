import theano.tensor as T
import numpy as np
import theano
from theano import shared


class rmsprop(object):
    """ RMSProp
    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.
    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.
    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:
    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}
    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    def __init__(self, params, masks=None, momentum=0.9):
        self.memory = []
        self.velocity = []
        for param in params:
            value = param.get_value(borrow=True)
            accu = shared(np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
            v = shared(np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
            self.memory.extend([accu])
            self.velocity.extend([v])
        self.masks = masks
        self.momentum = momentum

    def updates(self, params, grads, learning_rate,
                rho=0.9, epsilon=1e-6):
        """ RMSProp updates
        Parameters
        ----------
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        rho : float or symbolic scalar
            Gradient moving average decay factor
        epsilon : float or symbolic scalar
            Small value added for numerical stability
        Returns
        -------
        list(tuple)
            A tuple mapping each parameter to its update expression
        """

        one = T.constant(1)  # prevent upcasting of float32
        updates = []

        for n, (param, grad, mask) in enumerate(
                zip(params, grads, self.masks)):

            accu = self.memory[n]
            v = self.velocity[n]

            # update accu
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates.append((accu, accu_new))

            rmsprop = - (learning_rate * grad / T.sqrt(accu_new + epsilon))

            # update momentum
            v_new = self.momentum * v + rmsprop
            updates.append((v, v_new))

            update = self.momentum * v_new + rmsprop

            select = np.arange(len(mask.eval()))[mask.eval()]

            updates.append((param,
                T.inc_subtensor(param[select], update[select])))

        return updates
