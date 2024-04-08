import numpy as np
from plasma_MPC_get_solver import plasma_MPC_get_solver, dynamics
import matplotlib.pyplot as plt
import tikzplotlib
import time
import pickle


def plasma_MPC_get_samples(solver, X_feed, lb_relaxed, ub_relaxed, con_lb, con_ub, n, m, N):
    u_list = np.zeros((X_feed.shape[0], m))
    infeasible_points = np.zeros((X_feed.shape[0]))

    y_opt_nom = np.array((list(X_feed[0, :])*(N+1)+[4]*N*m+[0]))
    if np.size(X_feed) > 0 and X_feed.shape[1] != n:
        raise ValueError('Error with state values!')

    for i_iter in range(X_feed.shape[0]):
        # print(i_iter)
        lb_relaxed[:n] = X_feed[i_iter]
        ub_relaxed[:n] = X_feed[i_iter]
        res = solver(x0=y_opt_nom,
                     lbx=lb_relaxed,
                     ubx=ub_relaxed,
                     lbg=con_lb,
                     ubg=con_ub)
        y_opt_nom = res['x'].full()  # "warm start"
        # print(res['f'].full())
        if y_opt_nom[-1] < -1e-6:
            print(f'Negative slack variable at: {X_feed[i_iter, :]}')
        if y_opt_nom[-1] > 1e-10:
            infeasible_points[i_iter] = 1
        else:
            infeasible_points[i_iter] = 0
        u_MPC = y_opt_nom[n * (N + 1):n * (N + 1) + m].flatten()
        u_list[i_iter, :] = u_MPC
    return u_list, infeasible_points


def closed_loop_MPC(num_iter, solver, lb_relaxed, ub_relaxed, con_lb, con_ub, n, m, N, x=np.array([35, 58, 0])):
    X = np.zeros([num_iter+1, 3])
    X[0, :] = x.flatten()
    y_opt_nom = np.zeros(604)
    u_list = np.zeros([num_iter, 2])
    for i_iter in range(num_iter):
        lb_relaxed[:n] = X[i_iter, :]
        ub_relaxed[:n] = X[i_iter, :]
        res = solver(x0=y_opt_nom,
                     lbx=lb_relaxed,
                     ubx=ub_relaxed,
                     lbg=con_lb,
                     ubg=con_ub)
        y_opt_nom = res['x'].full()
        print(res['f'].full())
        u_MPC = y_opt_nom[n * (N + 1):n * (N + 1) + m].flatten()
        u_list[i_iter, :] = u_MPC
        x = dynamics(x, u_MPC, delta_t=0.5)
        X[i_iter+1, 0] = float(x[0])
        X[i_iter+1, 1] = float(x[1])
        X[i_iter+1, 2] = float(x[2])
    return X, u_list

