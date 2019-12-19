import numpy as np

np.random.seed(101)


def third_degree_poly(x):
    return x ** 3 - 4 * x ** 2 + 2 * x - 5


def third_degree_poly_with_noise(x, noise_scale):
    return x ** 3 - 4 * x ** 2 + 2 * x - 5 + np.random.rand(x.shape[0]) * noise_scale
