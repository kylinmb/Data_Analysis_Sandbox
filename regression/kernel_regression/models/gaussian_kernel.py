import numpy as np
from scipy.stats import norm


# This function uses the Nadaraya-Watson method to perform
# kernel regression using a gaussian kernel
def predict(x_new, x_train, y_train, sigma):
    y_predicted = np.zeros(x_new.shape[0])
    for x_n_index, x_n in enumerate(x_new):
        numerator = 0
        denominator = 0
        for x_t_index, x_t in enumerate(x_train):
            numerator += norm.pdf(x_n, x_t, sigma) * y_train[x_t_index]
            denominator += norm.pdf(x_n, x_t, sigma)
        y_predicted[x_n_index] = numerator / denominator
    return y_predicted


