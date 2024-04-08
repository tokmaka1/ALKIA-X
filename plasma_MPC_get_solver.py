import numpy as np
from casadi import *


def plasma_MPC_get_solver():
    n = 3  # state dimension
    m = 2  # input dimension
    CEM_sp = 10  # we want 10 mins of treatment

    T = 60  # 1 minute continuous horizon time
    delta_t = 0.5  # 0.5s sampling time
    N = np.ceil(T / delta_t).astype(int)  # discrete horizon
    penalty_weight = 10**6  # weight of the slack variable on the cost

    y = MX.sym('y', N*m+(N+1)*n+1)  # the +1 is for the slack variable sigma
    my_sigma = y[-1]

    x1min, x1max = 25, 42.5
    x2min, x2max = 20, 80
    x3min, x3max = 0, 11
    u1min, u1max = 1.5, 8
    u2min, u2max = 1, 6
    lb_x = np.array([x1min, x2min, x3min])
    lb_u = np.array([u1min, u2min])
    ub_x = np.array([x1max, x2max, x3max])
    ub_u = np.array([u1max, u2max])

    lb_relaxed = -np.infty * np.ones(n*(N+1)+m*N+1)  # np.zeros(n*(N+1)+m*N+1)
    ub_relaxed = np.infty * np.ones(n*(N+1)+m*N+1)  # last one is for sigma
    for i in range(N):
        lb_relaxed[n*(N+1)+m*i:n*(N+1)+m*(i+1)] = lb_u  # we do not relaxed input constraints! only state
        ub_relaxed[n*(N+1)+m*i:n*(N+1)+m*(i+1)] = ub_u
    lb_relaxed[-1] = 0  # lower bound on the slack variable sigma!

    con_eq = equality_constraints(y, delta_t, N, n, m)  # looks ok compared to matlab

    con_bound = np.zeros(N*n)
    con_lb = np.concatenate((con_bound, -np.inf * np.ones(n*(N+1)*2)))
    con_ub = np.concatenate((con_bound, np.zeros(n*(N+1)*2)))

    states = y[:n * (N + 1)]
    inputs = y[n * (N + 1):n * (N + 1) + N * m]

    con_new = con_eq
    for k in range(N+1):  # maybe there is something wrong with indices... But not so probable
        con_new = vertcat(con_new, states[k*n:(k+1)*n]-ub_x-my_sigma*np.ones((n, 1)))
        con_new = vertcat(con_new, -states[k*n:(k+1)*n]+lb_x-my_sigma*np.ones((n, 1)))

    obj = MX(0)
    obj = costfunction(y, n, m, N, CEM_sp, penalty_weight)

    options = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-8, 'jit': True, 'ipopt.max_iter': 100}  # plasma
    # options = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-11, 'jit': True, 'ipopt.max_iter': 100}  # CSTR

    nlp = {'x': y, 'f': obj, 'g': con_new}
    solver = nlpsol('solver', 'ipopt', nlp, options)

    return solver, lb_relaxed, ub_relaxed, con_lb, con_ub, n, m, N


def equality_constraints(y, delta_t, N, n, m):
    x = y[:n*(N+1)]
    u = y[n*(N+1):n*(N+1)+N*m]
    ceq = []

    for k in range(N):
        x_k = x[k*n:(k+1)*n]
        x_new = x[(k+1)*n:(k+2)*n]  # wrong?
        u_k = u[k*m:(k+1)*m]
        ceqnew = x_new - dynamics(x_k, u_k, delta_t)
        ceq = vertcat(ceq, ceqnew)

    return ceq

def dynamics(x, u, delta_t):
    xs = [38, 43.5, 0]
    us = [3, 3]
    x1, x2, x3 = x[0], x[1], x[2]
    u1, u2 = u[0], u[1]
    K = 0.5
    x1_plus = xs[0] + 0.42 * (x1 - xs[0]) + 0.68 * (x2 - xs[1]) + 1.58 * (u1 - us[0]) - 1.02 * (u2 - us[1])
    x2_plus = xs[1] - 0.06 * (x1 - xs[0]) + 0.26 * (x2 - xs[1]) + 0.73 * (u1 - us[0]) + 0.03 * (u2 - us[1])
    x3_plus = x3 + (K ** (43 - x1)) * delta_t
    xplus = vertcat(x1_plus, x2_plus, x3_plus)
    return xplus


def costfunction(y, n, m, N, CEM_sp, penalty_weight):
    my_sigma = y[-1]
    # x = y[:n*N]
    x = y[:n*(N+1)]
    CEM_N = x[n-1::n]  # Every CEM value.
    cost = 1e-2*norm_2(CEM_N - CEM_sp)**2 + (CEM_N[-1] - CEM_sp)**2 + (penalty_weight) * my_sigma
    return cost
