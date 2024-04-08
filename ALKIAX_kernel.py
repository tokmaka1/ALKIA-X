from sklearn.gaussian_process.kernels import Matern
from ALKIAX_gpr import ALKIAX_GaussianProcessRegressor


class matern_kernel():
    def __init__(self, sigma, ell, nu):
        self.sigma = sigma  # output variance
        self.ell = ell  # length scale
        self.nu = nu  # smoothness parameter
        self.kernel = self.sigma * Matern(length_scale=ell, length_scale_bounds='fixed', nu=self.nu)
        self.gaussian_process = ALKIAX_GaussianProcessRegressor(kernel=self.kernel)
