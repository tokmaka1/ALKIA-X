'''
Author: Abdullah Tokmak (abdullah.tokmak@aalto.fi)
Date: 15.01.2024
Modified version of gpr.py from sklearn to fit ALKIA-X needs
'''
from sklearn.gaussian_process._gpr import GaussianProcessRegressor
import warnings
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular

GPR_CHOLESKY_LOWER = True


class ALKIAX_GaussianProcessRegressor(GaussianProcessRegressor):
    # Modified version of the GaussianProcessRegressor class from sklearn
    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
    def __init__(
        self,
        kernel,
        *,
        optimizer=None,
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=False,
        random_state=None,
    ):
        self.kernel = kernel
        self.alpha = 0  # no regularization needed
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def covariance_matrix(self, X):
        '''
        Args:
            X (array): samples
        Returns:
            K (array): covariance matrix given samples X
            L (array): lower triangular matrix Cholesky decomposition
        '''
        K = self.kernel(X)
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                f"The kernel, {self.kernel_}, is not returning a positive "
                "definite matrix. Try gradually increasing the 'alpha' "
                "parameter of your GaussianProcessRegressor estimator.",
            ) + exc.args
            raise
        return K, L  # only return K and its cholesky decomposition

    def fit(self, X, y):
        '''
        Args:
            X (array): samples
            y (array): output corresponding to samples, i.e., y=f(X)
        Returns:
            coeffs (array): cofficients of RKHS function
        '''
        _, L = self.covariance_matrix(X)
        coeffs = cho_solve((L, GPR_CHOLESKY_LOWER), y, check_finite=False)
        return coeffs

    def power_function(self):
        '''
        Args:
            []
        Returns:
            power function (function): power function that can be evaluated with inputs X_train and X_test
        '''
        def power_function_inner(X_train, X_test):
            '''
            Args:
                X_train (array): given samples, with which we "trained" the model/built the approx. function
                X_test (arrray): samples where to evaluate power function
            Return:
                power function (function)
            '''
            _, L = self.covariance_matrix(X_train)
            K_trans = self.kernel(X_test, X_train)  # covariance vector
            V = solve_triangular(L, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False)
            y_var = self.kernel.diag(X_test)
            y_var -= np.einsum("ij,ji->i", V.T, V)
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn(
                    "Predicted variances smaller than 0. "
                    "Setting those variances to 0."
                )
                y_var[y_var_negative] = 0.0

            if len(y_var) != 1:  # because the power function is just one value
                y_var = y_var[0]
            power_function = np.sqrt(y_var)
            return power_function
        return lambda X_train, X_test: power_function_inner(X_train, X_test)

    def predict(self, X_train, coeffs):
        '''
        Args:
            X_train (array): samples with which RKHS function is built
            coeffs (array): coefficients of the RKHS function
        Return:
            prediction function (function): prediction function that can be evaluated at input X
        '''
        return lambda X: self.kernel(X.reshape(1, X_train.shape[1]), X_train) @ coeffs  # TODO: do not hard-code dimensions, it should work for all!

    def RKHS_norm(self, fX, coeffs):
        '''
        Args:
            fX (array): function evaluations at samples X
            coeffs (array): coefficients of RKHS function corresponding to inputs X
        '''
        h_RKHS_squared = fX.T @ coeffs  # correct since coeffs=K_inv@fX
        if h_RKHS_squared.ndim > 1 and h_RKHS_squared.shape[1] == 1:
            h_RKHS_squared = np.squeeze(h_RKHS_squared, axis=1)
        return np.sqrt(h_RKHS_squared)
