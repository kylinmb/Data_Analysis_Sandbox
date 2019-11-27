from data.toy_data import third_degree_poly_with_noise
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

np.random.seed(101)

# Generate data
x_train = np.linspace(start=-2, stop=5, num=20)
y_train = third_degree_poly_with_noise(x_train, 7)

x_test = np.linspace(start=-1.00, stop=5, num=10)
y_test = third_degree_poly_with_noise(x_test, 7)


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


# Predict value for test data and computes MSE
y_predicted = predict(x_test, x_train, y_train, 0.25)
mse = mean_squared_error(y_test, y_predicted)

# Plot Ground Truth and Predicted
plt.plot(x_test, y_test, label='Ground Truth')
plt.plot(x_test, y_predicted, label='Predicted')
plt.xlabel('Input (x)')
plt.ylabel('Ground Truth and Predicted Output')
plt.title('Ground Truth vs Predicted\nMSE ={0:.2f}'.format(mse))
plt.legend()
plt.show()
