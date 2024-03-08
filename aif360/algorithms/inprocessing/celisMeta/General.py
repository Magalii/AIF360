from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score


class General(ABC):
    """This is the class with the general functions of the algorithm.

    For different fairness metrics, the objective function of the optimization
    problem is different and hence needs different implementations.
    The fairness-metric specific methods need to extend this class and implement
    the necessary functions.
    """
    @abstractmethod
    def getExpectedGrad(self, dist, a, b, params, samples, mu, z_prior):
        """Used in gradient descent algorithm. Returns the value of gradient at
        any step.
        """
        raise NotImplementedError

    @abstractmethod
    def getValueForX(self, dist, a, b, params, z_prior, x):
        """Returns the threshold value at any point."""
        raise NotImplementedError

    @abstractmethod
    def getFuncValue(self, dist, a, b, params, samples, z_prior):
        """Returns the value of the objective function for given parameters."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_params(self):
        raise NotImplementedError

    #performs steps 2 of algo 1 in Celis et al
    #chosen such that algo considers ratio metrics
    def range(self, eps, tau):
        a = np.arange(np.ceil(tau/eps), step=10) * eps
        b = (a + eps) / tau
        b = np.minimum(b, 1)
        return np.c_[a, b]

    @abstractmethod
    def gamma(self, y_true, y_pred, sens):
        raise NotImplementedError

    def init_params(self, i):
        return [i] * self.num_params

    #dist = multivariate_normal(mean, cov, allow_singular=True, seed=random_state)
    def gradientDescent(self, dist, a, b, samples, z_prior):
        """Gradient Descent implementation for the optimizing the objective
        function.

        Note that one can alternately also use packages like CVXPY here.
        Here we use decaying step size. For certain objectives, constant step
        size might be better.
        """
        min_val = np.inf  # 1e8
        min_param = None
        for i in range(1, 10):
            params = self.init_params(i)
            for k in range(1, 50):
                grad = self.getExpectedGrad(dist, a, b, params, samples, 0.01,
                                            z_prior)
                for j in range(self.num_params):
                    params[j] = params[j] - 1/k * grad[j]
                f_val = self.getFuncValue(dist, a, b, params, samples, z_prior)
                if f_val < min_val:
                    min_val, min_param = f_val, params
        return min_param

    def prob(self, dist, x):
        return dist.pdf(x)

    #getModel(self.tau, x_train, y_train, x_control_train, self.seed)
    # tau : fairness parameter
    # X: features of training set (x_train)
    # y: labels of training set (y_train)
    # x_control_train: boolean array indicating in which positions elements belong to priviledged group
    def getModel(self, tau, X, y, sens, random_state=None):
        """Returns the model given the training data and input tau."""
        
        #1. Compute an estimated distribution dist on X,y,sens
        train = np.c_[X, y, sens] #Translates slice objects to concatenation along the second axis [[X_1, y_1, sens_1],...,[X_n, y_n, sens_n]]
        mean = np.mean(train, axis=0) #1D array
        cov = np.cov(train, rowvar=False) #Covariance matrix
        dist = multivariate_normal(mean, cov, allow_singular=True,
                                   seed=random_state) #multivariate normal random variable with fixed mean and covariance based on X, y and sens
        n = X.shape[1]
        dist_x = multivariate_normal(mean[:n], cov[:n, :n], allow_singular=True,
                                     seed=random_state) #multivariate normal random variable with fixed mean and covariance based on X training features only

        eps = 0.01 #error parameter
        z_1 = np.mean(sens)
        params_opt = [0] * self.num_params #array of size num_params, defined for each metric type
        max_acc = 0
        p, q = 0, 0

        if tau != 0:
            for a, b in self.range(eps, tau): #2. T <- /tau/eps\. For each a_i <- (i-1)*eps and b_i <- i*eps/tau
                samples = dist_x.rvs(size=1000) # TODO: why 20?  #Draw random samples from the multivariate normal distribution based on features
                                                # increasing sample size reduces standard variation and increases mean in accuracy (by reducing events of bad accuracy)
                params = self.gradientDescent(dist, a, b, samples, z_1)

                t = self.getValueForX(dist, a, b, params, z_1, X) #Probably impact for ratio instead of difference
                y_pred = np.where(t > 0, 1, -1)

                acc = accuracy_score(y, y_pred)
                gamma = self.gamma(y, y_pred, sens)

                if max_acc < acc and gamma >= tau - 0.2:  # TODO: why - 0.2?
                    max_acc = acc
                    params_opt = params
                    p, q = a, b
        return partial(self.getValueForX, dist, p, q, params_opt, z_1)
