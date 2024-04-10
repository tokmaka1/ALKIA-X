import numpy as np
from plasma_MPC_get_solver import plasma_MPC_get_solver
from plasma_MPC_get_samples import plasma_MPC_get_samples
from CSTR_MPC_get_solver import CSTR_MPC_get_solver
from CSTR_MPC_get_samples import CSTR_MPC_get_samples


class plasma_python():
    def __init__(self):
        self.type = 'MPC Python'
        self.x_dim = 3
        self.y_dim = 2

    def get_solver(self):
        self.solver, self.lb_relaxed, self.ub_relaxed, self.con_lb, self.con_ub, self.n, self.m, self.N = plasma_MPC_get_solver()

    def get_function_values(self, X_feed_normalized):
        # Re-shift our input here
        # definition area of the plasma MPC example:
        # x1 \in [25, 42.5]
        # x2 \in [20, 80]
        # x3 \in [0, 11]
        X_feed_reshaped = X_feed_normalized.copy()
        x_begin = [25, 20, 0]
        x_end = [42.5, 80, 11]
        for i in range(X_feed_reshaped.shape[0]):
            for k in range(X_feed_reshaped.shape[1]):
                X_feed_reshaped[i, k] = X_feed_reshaped[i, k]*x_end[k] + (1-X_feed_reshaped[i, k])*x_begin[k]
        u_list, infeasible_points = plasma_MPC_get_samples(self.solver, X_feed_reshaped, self.lb_relaxed, self.ub_relaxed, self.con_lb, self.con_ub, self.n, self.m, self.N)
        return u_list, infeasible_points


class cstr_python():
    def __init__(self):
        self.type = 'MPC Python'
        self.x_dim = 2
        self.y_dim = 1

    def get_solver(self):
        self.solver, self.lb_relaxed, self.ub_relaxed, self.con_lb, self.con_ub, self.N, self.m, self.n = RMPC_get_solver()

    def get_function_values(self, X_feed_normalized):
        # Re-shift our input here
        # definition area of the plasma MPC example:
        # x1 \in [-0.2, 0.2]
        # x2 \in [-0.2, 0.2]
        X_feed_reshaped = X_feed_normalized.copy()
        x_begin = [-0.2, -0.2]
        x_end = [0.2, 0.2]
        for i in range(X_feed_reshaped.shape[0]):
            for k in range(X_feed_reshaped.shape[1]):
                X_feed_reshaped[i, k] = X_feed_reshaped[i, k]*x_end[k] + (1-X_feed_reshaped[i, k])*x_begin[k]
        u_list, infeasible_points = RMPC_get_samples(self.solver, X_feed_reshaped, self.lb_relaxed, self.ub_relaxed, self.con_lb, self.con_ub, self.N, self.m, self.n)
        return u_list, infeasible_points


class waterfall_2D():
    def __init__(self):
        self.type = 'toy_example'
        self.x_dim = 2
        self.y_dim = 1

    def get_function_values(self, X):
        fX = np.zeros_like(X[:, 0])
        for i in range(len(fX)):
            if X[i, 0] < 0.48:
                fX[i] = 1 + 0.1*(np.sin(2*np.pi*X[i, 0]) + np.cos(2*np.pi*X[i, 1]))
            elif X[i, 0] < 0.52:
                fX[i] = (-50*X[i, 0] + 25) + 0.1*(np.sin(2*np.pi*X[i, 0]) + np.cos(2*np.pi*X[i, 1]))
            else:
                fX[i] = -1 + 0.1*(np.sin(2*np.pi*X[i, 0]) + np.cos(2*np.pi*X[i, 1]))
        infeasible_points = np.zeros_like(fX).astype(int)
        return fX, infeasible_points


class sine_2D(waterfall_2D):
    def get_function_values(self, X):
        fX = np.sin(2*np.pi*X[:, 0]) + np.cos(2*np.pi*X[:, 1]) \
            if len(np.shape(X)) > 1 else np.sin(2*np.pi*X[0]) + np.cos(2*np.pi*X[1])
        infeasible_points = np.zeros_like(fX).astype(int)
        return np.array(fX), infeasible_points


class sine_3D(waterfall_2D):
    def __init__(self):
        super().__init__()
        self.x_dim = 3
        self.y_dim = 2

    def get_function_values(self, X):
        fX = np.sin(2*np.pi*X[:, 0]) + np.cos(2*np.pi*X[:, 1]) \
            + 0.5*(np.cos(2*np.pi*X[:, 2]) + np.sin(2*np.pi*X[:, 2])) \
            if len(np.shape(X)) > 1 else np.sin(2*np.pi*X[0]) \
            + np.cos(2*np.pi*X[1]) + 0.5*(np.sin(2*np.pi*X[2]) + np.cos(2*np.pi*X[2]))
        infeasible_points = np.zeros_like(fX).astype(int)
        return np.array([fX, fX + 1]).T, infeasible_points


def ground_truth(string, eng=None):
    if string == 'plasma_python':
        return plasma_python()
    if string == 'CSTR_python':
        return cstr_python()
    if string == 'waterfall_2D':
        return waterfall_2D()
    elif string == 'sine_2D':
        return sine_2D()
    elif string == 'sine_3D':
        return sine_3D()


def ground_truth_dimensions(string):
    if string == 'waterfall_2D' or string == 'sine_2D' or string == 'CSTR_python':
        x_dim = 2
        y_dim = 1
    if string == 'sine_3D' or string == 'plasma' or string == 'plasma_python':
        x_dim = 3
        y_dim = 2
    return x_dim, y_dim


def ground_truth_type(string):
    if string == 'waterfall_2D' or string == 'sine_2D' or string == 'sine_3D':
        gt_type = 'toy_example'
    elif string == 'CSTR' or string == 'plasma':
        gt_type = 'MPC'
    elif string == 'plasma_python' or 'CSTR_python':
        gt_type = 'MPC Python'
    return gt_type
