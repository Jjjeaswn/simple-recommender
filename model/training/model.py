import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


class Model(object):
    def __init__(self, theta=None, latent_dim=1, learning_rate=.05, lambda_weight=.0001, losses_log=True):

        if theta is None:
            if latent_dim is not None:
                theta = np.random.random(latent_dim) * 0.1

        self.theta = np.array(theta).astype(float)
        self._alpha = learning_rate
        self._lambda = lambda_weight

        self.epochs = None

        self.x = None
        self.y = None

        self.losses_log = losses_log
        self.losses = []

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
            self.theta = np.random.random(kwargs['latent_dim'])
        return self

    def fit(self, x, y, epochs=50, learning_rate=None, lambda_weight=None):
        """

        Args:
            y: user's item ratings
            epochs: total times we train the model
            lambda_weight:
            learning_rate:
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
        # if loss > 100:
        #     self.theta = np.random.random(self.theta.shape) * 0.1
        # else:
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


if __name__ == '__main__':
    model = Model(latent_dim=4)
    # x = np.random.random((10, 10))
    x = np.array([
        [1, .9, 0, 0],
        [1, 0, .9, 0],
        [1, 0, 0, .9]
    ])
    # x[0] = 1.
    # y = np.random.randint(1, 6, 6).astype(float)
    y = np.array([
        5.,
        1.,
        3.
    ])

    clf = GridSearchCV(model, {
        'learning_rate': [.01, .05, .001, .005],
        'lambda_weight': [.1, .01, .001, .0001],
        'latent_dim': [4]
    })

    clf.fit(x, y)
    print('best score = ', clf.best_score_)

    print(clf.best_estimator_.predict(x))
    print(y)
    print(clf.best_estimator_.theta)

    clf.best_estimator_.plot_losses()
