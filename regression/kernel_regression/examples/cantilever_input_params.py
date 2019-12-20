"""
This example uses the kernel regression model to generate new
input parameters from QoIs
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from regression.kernel_regression.models.gaussian_kernel import predict

# So results are reproducible
np.random.seed(101)

# Settings - You can adjust these to look at different data sets, QoIs, input params, etc.
persistence_level = 17
crystal_id = 2
number_of_steps = 50
sigma = 0.25
qoi_name = 'Max Stress'
input_name = 'Angle'
# Path settings
path = '/usr/sci/projects/dSpaceX/DATA/CantileverBeam_wclust_wraw/'
images_path = '../generated_figures/'
input_filename = 'CantileverBeam_design_parameters.csv'
qoi_filename = 'CantileverBeam_QoIs.csv'
crystal_filename = 'CantileverBeam_CrystalPartitions_maxStress.csv'

# Get input parameters, QOIS, and crystal partitions from file
input_df = pd.read_csv(path + input_filename)
qoi_df = pd.read_csv(path + qoi_filename)
crystal_df = pd.read_csv(path + crystal_filename, header=None)

# Get samples belonging to specified crystal
crystal = crystal_df.loc[persistence_level, crystal_df.loc[persistence_level, :] == crystal_id].index.to_numpy()

# QOI (x) and  Parameter (y) for persistence level and single crystal
qoi_df_crystal = qoi_df.loc[crystal, :]
param_df_crystal = input_df.loc[crystal, :]

# Create general training and test data sets
mask = np.random.rand(len(qoi_df_crystal)) < 0.7
qoi_train_df = qoi_df_crystal[mask]
qoi_test_df = qoi_df_crystal[~mask]

param_train_df = param_df_crystal[mask]
param_test_df = param_df_crystal[~mask]

# Select single qoi of interest and create numpy arrays
train_df = pd.concat([qoi_train_df, param_train_df], axis=1).sort_values(by=[qoi_name])
train_x = train_df.loc[:, qoi_name].to_numpy().reshape((-1, 1))
train_y = train_df.loc[:, input_name].to_numpy().reshape((-1, 1))

test_df = pd.concat([qoi_test_df, param_test_df], axis=1).sort_values(by=[qoi_name])
test_x = test_df.loc[:, qoi_name].to_numpy().reshape((-1, 1))
test_y = test_df.loc[:, input_name].to_numpy().reshape((-1, 1))

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
fig.savefig(images_path + 'GroundTruth_Prediction_PL'+str(persistence_level)+'_CID'+str(crystal_id)+'_QOI'+qoi_name+
            '_IN'+input_name+'.png')
plt.show()

# Now generate 50 samples between qoi min and max
min_qoi = qoi_df_crystal.loc[:, qoi_name].min()
max_qoi = qoi_df_crystal.loc[:, qoi_name].max()
step_size = (max_qoi - min_qoi) / number_of_steps
new_samples = np.arange(min_qoi, max_qoi, step_size).reshape((-1, 1))

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
fig.savefig(images_path + 'Training_Data_Prediction_PL'+str(persistence_level)+'_CID'+str(crystal_id)+'_QOI'+qoi_name+
            '_IN'+input_name+'.png')
plt.show()

# Plot Prediction Only
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(new_samples, new_predicted)
ax.set_xlabel(qoi_name)
ax.set_ylabel(input_name)
ax.set_title('Predicted Value for 50 Equidistant Samples')
fig.savefig(images_path + 'Prediction_Only_PL'+str(persistence_level)+'_CID'+str(crystal_id)+'_QOI'+qoi_name+
            '_IN'+input_name+'.png')
plt.show()
