import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from regression.kernel_regression.models.gaussian_kernel import predict

np.random.seed(101)

# Settings
path = '../data/'
images_path = '../../../figures/'
input_filename = 'CantileverBeam_design_parameters.csv'
qoi_filename = 'CantileverBeam_QoIs.csv'
crystal_filename = 'crystalpartitions_truss_maxStress_server_12_18_2019_sigma15_persistence99.csv'
qoi_name = 'Max Stress'
input_name = 'Angle'
persistence_level = 17
crystal_id = 2
number_of_steps = 50
sigma = 0.25

# Get input parameters, QOIS, and crystal partitions from file
input_df = pd.read_csv(path + input_filename)
qoi_df = pd.read_csv(path + qoi_filename)
crystal_df = pd.read_csv(path + crystal_filename, header=None)

# Get samples belonging to specified crystal
crystal = crystal_df.loc[persistence_level, crystal_df.loc[persistence_level, :] == crystal_id].index.to_numpy()

# QOI (x) and  Parameter (y) for persistence level and single crystal
qoi_df_crystal = qoi_df.loc[crystal, :]
param_df_crystal = input_df.loc[crystal, :]

# Create training and test data set for specific qoi and input param of interest
mask = np.random.rand(len(qoi_df_crystal)) < 0.7
qoi_train_df = qoi_df_crystal[mask]
qoi_test_df = qoi_df_crystal[~mask]

param_train_df = param_df_crystal[mask]
param_test_df = param_df_crystal[~mask]

train_df = pd.concat([qoi_train_df, param_train_df], axis=1).sort_values(by=[qoi_name])
train_x = train_df.loc[:, qoi_name].to_numpy()
train_y = train_df.loc[:, input_name].to_numpy()

test_df = pd.concat([qoi_test_df, param_test_df], axis=1).sort_values(by=[qoi_name])
test_x = test_df.loc[:, qoi_name].to_numpy()
test_y = test_df.loc[:, input_name].to_numpy()

# Predict value for test data and computes MSE
test_predicted = predict(test_x, train_x, train_y, sigma)
mse = mean_squared_error(test_y, test_predicted)

# Plot Ground Truth and Predicted
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(test_x, test_y, label='Ground Truth')
ax.plot(test_x, test_predicted, label='Predicted')
ax.set_xlabel(qoi_name)
ax.set_ylabel(input_name)
ax.set_title('Ground Truth vs Predicted for Persistence Level '
             + str(persistence_level) + ' and Crystal ' + str(crystal_id) + '\nMSE ={0:.2f}'.format(mse))
ax.legend()
fig.savefig()
plt.show()

# Now generate 50 samples between qoi min and max
min_qoi = qoi_df_crystal.loc[:, qoi_name].min()
max_qoi = qoi_df_crystal.loc[:, qoi_name].max()
step_size = (max_qoi - min_qoi) / number_of_steps
new_samples = np.arange(min_qoi, max_qoi, step_size)

# predict input param for new samples
new_predicted = predict(new_samples, train_x, train_y, sigma)

# Plot Prediction vs Training
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_x, train_y, label='Training Data')
ax.plot(new_samples, new_predicted, label='Predicted')
ax.set_xlabel(qoi_name)
ax.set_ylabel(input_name)
ax.set_title('Training Data vs Prediction for 50 Equidistant Samples')
ax.legend()
plt.show()

# Plot Prediction Only
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(new_samples, new_predicted)
ax.set_xlabel(qoi_name)
ax.set_ylabel(input_name)
ax.set_title('Predicted Value for 50 Equidistant Samples')
plt.show()