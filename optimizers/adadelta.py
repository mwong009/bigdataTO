from collections import OrderedDict
import theano.tensor as T
import numpy as np
import theano
from theano import shared


class adadelta(object):
    """ Adadelta
    Scale learning rates by the ratio of accumulated gradients to accumulated
    updates, see [1]_ and notes for further description.
    Notes
    -----
    rho should be between 0 and 1. A value of rho close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.
    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
    work for multiple datasets (MNIST, speech).
    In the paper, no learning rate is considered (so learning_rate=1.0).
    Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does
    not become 0).
    Using the step size eta and a decay factor rho the learning rate is
    calculated as:
    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                             {\sqrt{r_t + \epsilon}}\\\\
       s_t &= \\rho s_{t-1} + (1-\\rho)*(\\eta_t*g)^2
    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.
    """

    def __init__(self, params, masks=None, momentum=0.9):
        self.memory = []
        self.delta = []
        self.velocity = []
        for param in params:
            value = param.get_value(borrow=True)
            accu = shared(np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
            delta_accu = shared(np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
            v = shared(np.zeros(value.shape, dtype=value.dtype),
                broadcastable=param.broadcastable)
            self.memory.extend([accu])
            self.delta.extend([delta_accu])
            self.velocity.extend([v])
        self.masks = masks
        self.momentum = momentum

    def updates(self, params, grads, learning_rate, rho=0.9, epsilon=1e-6):
        """ Adadelta updates
        Parameters
        ----------
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        rho : float or symbolic scalar
            Squared gradient moving average decay factor
        epsilon : float or symbolic scalar
            Small value added for numerical stability
        Returns
        -------
        list(tuple)
            A tuple mapping each parameter to its update expression
        """

        one = T.constant(1)  # prevent upcasting of float32
        updates = OrderedDict()

        for n, (param, grad, mask) in enumerate(
            zip(params, grads, self.masks)):

            accu = self.memory[n]
            delta_accu = self.delta[n]
            v = self.velocity[n]

            # update accu
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates[accu] = accu_new

            # compute parameter update, using the 'old' delta_accu
            adadelta = - (learning_rate * (grad * T.sqrt(delta_accu + epsilon)
                / T.sqrt(accu_new + epsilon)))

            # update momentum
            v_new = self.momentum * v + adadelta
            updates[v] = v_new

            update = self.momentum * v_new + adadelta

            select = np.arange(len(mask.eval()))[mask.eval()]

            # compute parameter update, using the 'old' delta_accu
            updates[param] = T.inc_subtensor(param[select], update[select])

            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
            updates[delta_accu] = delta_accu_new

        return updates
