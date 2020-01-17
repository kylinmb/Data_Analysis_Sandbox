import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist


def predict(x_new, x_train, y_train, sigma):
    """
    Nadaraya-Watson Regression using Gaussian PDF for kernel
    Complete explanation can be found in: https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
    :param x_new: Input values you would like to make an output prediction for      // desired qois to create new z_coords 
    :param x_train: Known input values; used for training and prediction            // qoi
    :param y_train: Known output values; used for training and prediction           // z_coords for each sample's qoi
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


# matrix version of predict function above
def predict_matrix(x_new, x_train, y_train, sigma):
    """
    Uses the predict_matrix_helper function to perform Nadaraya-Watson Regression,
    a Gaussian PDF, and matrix algebra
    :param x_new: Input values you would like to make an output prediction for
    :param x_train: Known input values; used for training and prediction
    :param y_train: Known output values; used for training and prediction
    :param sigma: standard deviation for Gaussian PDF
    :return:
    """
    y_predicted = []
    for x in x_new:
        pred = predict_matrix_helper(x, x_train, y_train, sigma)
        y_predicted.append(pred)
    y_predicted = np.asarray(y_predicted)
    y_predicted = y_predicted.reshape(y_predicted.shape[0], y_predicted.shape[2])
    return y_predicted


# takes a single new point (ex: QoI) and calculates its new z_coord
def predict_matrix_helper(x_new, x_train, y_train, sigma):
    """
    Predict y value for a single input using the Nadaraya-Watson Regression,
    a Gaussian PDF, and matrix algebra
    :param x_new: Input values you would like to make an output prediction for
    :param x_train: Known input values; used for training and prediction
    :param y_train: Known output values; used for training and prediction
    :param sigma: standard deviation for Gaussian PDF
    :return:
    """
    # calculate difference
    difference = x_new - np.transpose(x_train)

    # Apply gaussian elementwise to difference:
    # G = [1 / sqrt(2*pi*sigma)] * e^(-1/2*sigma^2 * dist^2)
    difference_squared = np.square(difference)
    exponent = np.divide(difference_squared, -2*sigma**2)
    exponent = np.exp(exponent)
    denominator = np.sqrt(2*np.pi*sigma)
    gaussian_vector = exponent / denominator

    # calculate weight and normalization for regression
    summation = np.sum(gaussian_vector)
    output = gaussian_vector.dot(y_train)
    return output / summation


