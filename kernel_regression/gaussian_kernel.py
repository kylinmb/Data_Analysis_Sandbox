import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

np.random.seed(101)

# Third degree polynomial to generate toy data set
def third_degree_poly(x):
    return x**3 - 4 * x**2 + 2 * x - 5


# Create toy data set - simple third degree polynomial x^3 - 4x^2 + 2x - 5
x_train = np.linspace(start=-2, stop=5, num=20)
y_train = third_degree_poly(x_train) + np.random.rand(x_train.shape[0])*7

x_test = np.linspace(start=-1.00, stop=5, num=10)
y_test = third_degree_poly(x_test) + np.random.rand(x_test.shape[0])*7

plt.plot(x_train, y_train, 'o')
plt.show()


# Gaussian kernel
def kernel(x_1, x_2, sigma):
    k = np.subtract(x_1, x_2)
    k = - np.power(k, 2)
    k = k / (2 * sigma**2)
    return np.exp(k)


def predict(x_star, x_t, y_t, sigma):
    y_star = np.zeros(x_star.shape[0])
    for i, x_s in enumerate(x_star):
        numerator = 0
        denominator = 0
        individual_numerator = np.zeros(x_t.shape)
        individual_denominator = np.zeros(x_t.shape)
        for j, x in enumerate(x_t):
            individual_numerator[j] = norm.pdf(x_s, x, sigma) * y_t[j]
            individual_denominator[j] = norm.pdf(x_s, x, sigma)
            numerator += norm.pdf(x_s, x, sigma) * y_t[j]
            denominator += norm.pdf(x_s, x, sigma)
            # numerator += kernel(x_s, x, sigma) * y_t[i]
            # denominator += kernel(x_s, x, sigma)
        y_star[i] = numerator/denominator
    return y_star, individual_numerator, individual_denominator


y_predicted, indi_n, indi_d = predict(x_test, x_train, y_train, 1)
plt.plot(x_train, indi_n, 'r')
plt.show()
plt.plot(x_train, indi_d, 'b')
plt.show()
plt.plot(x_test, y_test)
plt.plot(x_test, y_predicted)
plt.show()
mse = mean_squared_error(y_test, y_predicted)
