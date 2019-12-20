import numpy as np
from scipy.stats import norm


def predict(x_new, x_train, y_train, sigma):
    """
    Nadaraya-Watson Regression using Gaussian PDF for kernel
    Complete explanation can be found in: https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
    :param x_new: Input values you would like to make an output prediction for
    :param x_train: Known input values; used for training and prediction
    :param y_train: Known output values; used for training and prediction
    :param sigma: standard deviation for Gaussian PDF
    :return: output prediction for x_new
    """
    # create container for predicted values - should have the same number of samples as x_new and the same
    # dimensions as y_train
    y_predicted = np.zeros((x_new.shape[0], y_train.shape[1]))
    # For each x_new we want to calculate the gaussian kernel between x_new and every x_train
    # and weight the contribution of y_train based on this value; the summation of these in the numerator.
    # We then normalize the numerator using the summation of just the kernel.
    for x_n_index, x_n in enumerate(x_new):
        numerator = 0
        denominator = 0
        for x_t_index, x_t in enumerate(x_train):
            numerator += norm.pdf(x_n, x_t, sigma) * y_train[x_t_index]
            denominator += norm.pdf(x_n, x_t, sigma)
        y_predicted[x_n_index] = numerator / denominator
    return y_predicted
