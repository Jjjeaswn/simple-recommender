import numpy as np
import matplotlib.pyplot as plt


class CFModel(object):
    def __init__(self,
                 theta=None,
                 latent_dim=1,
                 learning_rate=.05,
                 lambda_weight=.0001,
                 losses_log=True,
                 initial_theta_scale=.2,
                 early_stop_threshold=.1):
        """

        Args:
            theta (float):
            latent_dim (int):
            learning_rate (float):
            lambda_weight (float):
            losses_log (bool):
            initial_theta_scale(float):
            early_stop_threshold(float):
        """

        self.initial_theta_scale = initial_theta_scale
        self.early_stop_threshold = early_stop_threshold

        if theta is None:
            if latent_dim is not None:
                theta = np.random.random(latent_dim) * self.initial_theta_scale

        self.theta = np.array(theta, dtype=float)
        self._alpha = learning_rate
        self._lambda = lambda_weight

        self.epochs = None

        self.x = None
        self.y = None

        self.losses_log = losses_log
        self.losses = []  # type: list[float]

    def score(self, *args, **kwargs):
        score = 1 / np.sum(np.power(self.predict(self.x) - self.y, 2))
        print(score)
        return score

    def get_params(self, deep=True):
        return {
            'learning_rate': self._alpha,
            'lambda_weight': self._lambda
        }

    def set_params(self, **kwargs):
        self._alpha = kwargs['learning_rate']
        self._lambda = kwargs['lambda_weight']

        if kwargs['latent_dim'] is not None:
            self.theta = np.random.random(kwargs['latent_dim']) * self.initial_theta_scale
        return self

    def fit(self, x, y, epochs=50, learning_rate=None, lambda_weight=None):
        """

        Args:
            x(np.ndarray | list of list of float):
            y(np.ndarray | list):
            epochs:
            learning_rate:
            lambda_weight:

        Returns:

        """

        if learning_rate is not None:
            self._alpha = learning_rate

        if lambda_weight is not None:
            self._lambda = lambda_weight

        self.epochs = epochs
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)

        self.x[:, 0] = 1.

        for epoch in range(self.epochs):
            h = self.predict(self.x)
            self.backward(h)

    def predict(self, x):
        h = np.matmul(x, self.theta)
        assert h.shape == (x.shape[0],)
        return h

    def backward(self, h):
        loss = self.__compute_loss(h)
        if loss > 100:
            # loss explosion
            # simply re-random theta
            self.theta = np.random.random(self.theta.shape) * self.initial_theta_scale
        elif loss < self.early_stop_threshold:
            # stop tuning this theta
            pass
        else:
            self.__gradient_descent(h)

    def __gradient_descent(self, h):
        assert h.shape == self.y.shape

        diff = h - self.y

        assert diff.shape == (self.x.shape[0],)

        diff_sum = np.sum(self.x.T * diff, axis=1)

        assert diff_sum.shape == self.theta.shape

        theta = np.copy(self.theta)
        theta[0] = 0
        grads = diff_sum + self._lambda * theta
        assert grads.shape == self.theta.shape
        self.theta -= grads * self._alpha

    def __compute_loss(self, h):
        loss = np.sum(np.power(h - self.y, 2)) / 2 + np.sum(np.power(self.theta, 2)) * self._lambda / 2
        # print('Current loss = ', loss)
        if self.losses_log:
            self.losses.append(loss)
        return loss

    def plot_losses(self):
        plt.plot([i for i in range(len(self.losses))], self.losses)
        print('alpha = {}, lambda = {}'.format(self._alpha, self._lambda))
        plt.text(2, 2, 'alpha = {}, lambda = {}'.format(self._alpha, self._lambda))
        plt.show()
