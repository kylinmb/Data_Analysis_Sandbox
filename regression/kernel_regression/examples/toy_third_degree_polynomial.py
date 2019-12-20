"""
This example uses the kernel regression model to approximate
the third degree polynomial in toy_data.py
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from regression.kernel_regression.data.toy_data import third_degree_poly_with_noise
from regression.kernel_regression.models.gaussian_kernel import predict

np.random.seed(101)

# settings
images_path = '../generated_figures/'

# Get data
x_train = np.linspace(start=-2, stop=5, num=20).reshape((-1, 1))
y_train = third_degree_poly_with_noise(x_train, 7)

x_test = np.linspace(start=-1.00, stop=5, num=10).reshape((-1, 1))
y_test = third_degree_poly_with_noise(x_test, 7).reshape((-1, 1))

# Predict value for test data and computes MSE
y_predicted = predict(x_test, x_train, y_train, 0.25)
mse = mean_squared_error(y_test, y_predicted)

# Plot Ground Truth and Predicted
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_test, y_test, label='Ground Truth')
ax.plot(x_test, y_predicted, label='Predicted')
ax.set_xlabel('Input (x)')
ax.set_ylabel('Ground Truth and Predicted Output')
ax.set_title('Ground Truth vs Predicted\nMSE ={0:.2f}'.format(mse))
ax.legend()
fig.savefig(images_path + 'toy_third_degree_polynomial_predicted_vs_ground_truth.png')
plt.show()
