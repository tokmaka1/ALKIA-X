import numpy as np
from RMPC_get_solver import RMPC_get_solver


def RMPC_get_samples(solver, X_feed, lb_relaxed, ub_relaxed, con_lb, con_ub, N, m, n):
    u_list = np.zeros((X_feed.shape[0], 1))
    infeasible_points = np.zeros((X_feed.shape[0], 1))

    y_opt_nom = np.zeros((N*m + (N+1)*n + 1, 1))  # size of y

    if X_feed.shape[1] != 2:
        raise ValueError("Something went wrong with inserting the feed points into the RMPC script. We have to insert 2D points here. Does the dimension get lost in the Python code?")

    for i_iter in range(X_feed.shape[0]):
        # print(i_iter)
        x1 = X_feed[i_iter, 0]
        x2 = X_feed[i_iter, 1]
        lb_relaxed[:n] = [x1, x2]
        ub_relaxed[:n] = [x1, x2]
        y_init = y_opt_nom.flatten()
        res = solver(x0=y_init,
                     lbx=lb_relaxed,
                     ubx=ub_relaxed,
                     lbg=con_lb,
                     ubg=con_ub)
        y_opt_nom = res['x'].full().flatten()
        g_violation = max(np.max(res['g'] - con_ub), np.max(con_lb - res['g']))

        sigma_res = y_opt_nom[-1]
        if sigma_res < -1e-8:
            raise ValueError('Negative slack variable')  # This is then really a problem and a conceptual one: return error
        infeasible_point = 1 if sigma_res >= 1e-10 else 0
        u_MPC = y_opt_nom[n*(N+1):n*(N+1)+m]
        u_list[i_iter] = u_MPC
        infeasible_points[i_iter] = infeasible_point

    ue = 0.7853  # steady state input
    u_list += ue

    return u_list.flatten(), infeasible_points

if __name__ == '__main__':
    solver, lb_relaxed, ub_relaxed, con_lb, con_ub, N, m, n = RMPC_get_solver()
    rows = 289
    cols = 2

    # Define the ranges for x1 and x2
    x1_range = np.linspace(-0.2, 0.2, int(rows ** 0.5))
    x2_range = np.linspace(-0.2, 0.2, int(rows ** 0.5))

    # Initialize the array
    X_feed = np.zeros((rows, cols))

    # Populate the array
    for i, x1 in enumerate(x1_range):
        for j, x2 in enumerate(x2_range):
            index = i * len(x2_range) + j
            X_feed[index, 0] = x1
            X_feed[index, 1] = x2

    # print(X_feed)
    # RMPC_get_samples(solver, X_feed, lb_relaxed, ub_relaxed, con_lb, con_ub, N, m, n)
