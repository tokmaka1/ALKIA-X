import numpy as np
from casadi import *
from scipy.io import loadmat
# from RMPC_get_samples import RMPC_get_samples
import time

def RMPC_get_solver():
    # Define system dimensions

    n = 2
    m = 1
    us = 0.7583
    uzul = 0.0051
    epsilon = 0.0022
    rho = 0.9913
    sigma_cost = 10**5  # weight for slack variable cost
    P_inf = np.array([[33.2120, -3.6133], [-3.6133, 6.6498]])
    R = 1e-4
    Q = np.array([[1,0], [0,1]])
    alpha = 9.2592e-5

    # Set horizon
    T = 18
    delta = 0.1  # h_val
    N = int(np.ceil(T / delta))

    # Define constraints
    x1max = 1
    x1min = 0
    x2max = 1
    x2min = 0.5

    # Shift constraints by steady-state
    xs = np.array([0.2632, 0.6519])
    x_max = np.array([x1max, x2max]) - xs
    x_min = np.array([x1min, x2min]) - xs

    # Decision variables
    y = MX.sym("y", N * m + (N + 1) * n + 1)

    # Compute cost and constraints
    obj = 0
    obj = costfunction(y, n, N, m, R, Q, P_inf, sigma_cost=10**5)
    c, ceq = nonlinearconstraints_nom(y, n, N, m, P_inf, alpha, xs, us, delta)
    con = vertcat(c, ceq)

    con_bound = np.zeros(n*N)
    # add terminal constraints
    con_lb = np.concatenate(([-np.inf], con_bound, [-np.inf] * (y.shape[0] * 2-2)))
    con_ub = np.concatenate(([0], con_bound, np.zeros((y.shape[0] * 2 - 2))))

    # Set box constraints, including tightening
    uub = (2 - us) * (1 - uzul)
    ulb = (0 - us) * (1 - uzul)
    xub = np.array([0.2, 0.2])
    xlb = np.array([-0.2, -0.2])

    lb_u = np.zeros(N)
    ub_u = np.zeros_like(lb_u)
    lb_x = np.zeros(2*(N+1))  # 2D state
    ub_x = np.zeros_like(lb_x)
    for k in range(N):
        epsilon_k = epsilon*(1 - np.sqrt(rho**k))/(1 - np.sqrt(rho))
        lb_u[k] = (1-epsilon_k) * ulb
        ub_u[k] = (1-epsilon_k) * uub
        lb_x[2*k:2*k+2] = (1-epsilon_k) * xlb
        ub_x[2*k:2*k+2] = (1-epsilon_k) * xub
    epsilon_k = epsilon * (1 - np.sqrt(rho)**N) / (1 - np.sqrt(rho))
    lb_x[-2:] = ((1 - epsilon_k) * xlb)
    ub_x[-2:] = ((1 - epsilon_k) * xub)

    lb = np.concatenate([lb_x, lb_u])
    ub = np.concatenate([ub_x, ub_u])

    my_x = y[:n * (N + 1)]
    my_u = y[n * (N + 1):(n * (N + 1)) + (N * m)]
    my_sigma = y[-1]

    # con_new = vertcat([con, [my_x, my_u] - ub - my_sigma * np.ones(y.shape[0]-1),
    #                   [-my_x, -my_u] + lb - my_sigma * np.ones(y.shape[0]-1)]
    #                   )
    con_new = vertcat(con, vertcat(my_x, my_u) - ub - my_sigma * np.ones(y.shape[0]-1),
                    vertcat(-my_x, -my_u) + lb - my_sigma * np.ones(y.shape[0]-1)
                    )

    lb_relaxed = np.concatenate((-np.inf * np.ones(y.shape[0]-1), [0]))
    ub_relaxed = np.infty * np.ones(y.shape[0])

    options = {}
    options["ipopt.print_level"] = 0
    options["print_time"] = 0
    options["ipopt.tol"] = 1e-11

    # JIT options
    options["jit"] = True
    # options["compiler"] = 'clang'
    # options["jit_options"] = {}
    # options["jit_options"]["flags"] = '-O3'
    # options["jit_options"]["verbose"] = True

    nlp = {"x": y, "f": obj, "g": con_new}
    solver = nlpsol("solver", "ipopt", nlp, options)

    return solver, lb_relaxed, ub_relaxed, con_lb, con_ub, N, m, n


def costfunction(y, n, N, m, R, Q, P_inf, sigma_cost=10**5):
    x = y[:n*(N + 1)]
    u = y[n * (N+1):(n * (N + 1)) + (N * m)]
    sigma = y[-1]
    cost = 0

    for k in range(1, N + 1):
        x_k = x[(k - 1) * n:k * n]
        u_k = u[(k - 1) * m:k * m]
        cost += runningcosts(x_k, u_k, Q, R)

    x_N = x[n * (N + 1 - 1):n * (N + 1)]
    cost += runningcosts(x_N, u_k, P_inf, 0 * R)
    cost += sigma_cost * (sigma)

    return cost


def runningcosts(x, u, Q, R):
    return x.T @ Q @ x + u.T @ R @ u


def nonlinearconstraints_nom(y, n, N, m, P_inf, alpha, xs, us, h_val):
    x = y[:n * (N + 1)]
    u = y[n * (N + 1):(n * (N + 1)) + (N * m)]
    sigma = y[-1]
    c = []
    ceq = []

    for k in range(1, N + 1):
        x_k = x[(k - 1) * n:k * n]
        x_new = x[k * n:(k + 1) * n]
        u_k = u[(k - 1) * m:k * m]

        ceqnew = x_new - dynamic(x_k, u_k, xs, us, h_val)
        ceq = vertcat(ceq, ceqnew)

    x_N = x[n * (N + 1 - 1):n * (N + 1)]
    cnew = terminalconstraints(x_N, P_inf, alpha) - sigma
    c = vertcat(c, cnew)

    return c, ceq


def dynamic(x, u, xs, us, h_val):
    xs = np.array(xs)
    us = np.array(us)
    x1 = x[0] + xs[0]
    x2 = x[1] + xs[1]
    u = u + us

    # x2 = max(x2, 1e-4)
    theta = 20
    k = 300
    M = 5
    xf = 0.3947
    xc = 0.3816
    alpha = 0.117

    exp_expression = k * x1 * np.exp(-M / x2)
    f = [1 / theta * (1 - x1) - exp_expression, 1 / theta * (xf - x2) + exp_expression - alpha * u * (x2 - xc)]
    return x + h_val * vertcat(f[0], f[1])

def terminalconstraints(x, P_inf, alpha):
    return x.T @ P_inf @ x - alpha


if __name__ == '__main__':
    solver, lb_relaxed, ub_relaxed, con_lb, con_ub, N, m, n = RMPC_get_solver()
    M = 30
    D = np.array(list([[-0.2, 0.2]]*2))
    X_D = np.zeros((M**2, 2)) + D[:, 0]  # initialization
    print(f'Total number of points {X_D.shape[0]}')
    for number_of_point in range(X_D.shape[0]):
        x = X_D[number_of_point, :]
        rest_division = number_of_point
        exp = 2
        entry_of_point = 0
        while rest_division != 0:
            # "binary" logic of numbering
            exp = exp - 1
            int_division = int(rest_division/M**exp)
            x[entry_of_point] = D[entry_of_point][0] + (D[entry_of_point][1] - D[entry_of_point][0])/(M-1)*int_division
            rest_division = rest_division % (M**exp)
            entry_of_point += 1
        X_D[number_of_point] = x
    current_time = time.time()
    u_list, infeasible_points = RMPC_get_samples(solver, X_D, lb_relaxed, ub_relaxed, con_lb, con_ub, N, m, n)
    print(f'Time needed to solve: {time.time()-current_time}')
