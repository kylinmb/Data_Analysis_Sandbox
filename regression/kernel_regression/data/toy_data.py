import numpy as np

# So data will be reproducible
np.random.seed(101)


def third_degree_poly(x):
    """
    Calculates third degree polynomial
    x^2 - 4x^2 + 2x - 5
    :param x: input
    :return: value of polynomial
    """
    return x ** 3 - 4 * x ** 2 + 2 * x - 5


def third_degree_poly_with_noise(x, noise_scale):
    """
    Calculate thir degree polynomial with noise
    x^2 - 4x^2 + 2x - 5
    :param x: input
    :param noise_scale: scaling factor/magnitude of noise
    :return: value of polynomial with addition of noise
    """
    poly = third_degree_poly(x)
    noise = np.random.rand(x.shape[0], x.shape[1]) * noise_scale
    return poly + noise
