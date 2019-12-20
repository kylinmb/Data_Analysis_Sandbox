"""
This example uses the kernel regression model to generate new
latent space values from QoIs
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from regression.kernel_regression.models.gaussian_kernel import predict

# So results are reproducible
np.random.seed(101)

# Settings
persistence_level = 17
crystal_id = 2
qoi_name = 'Max Stress'
sigma = 0.25
number_of_steps = 50

# Path Settings
generated_data_path = '../generated_data/'
base_path = '/usr/sci/projects/dSpaceX/DATA/CantileverBeam_wclust_wraw/'
shape_odds_path = (base_path + 'shapeodds_models_maxStress/persistence-' + str(persistence_level)
                   + '/crystal-' + str(crystal_id) + '/')
latent_csv = shape_odds_path + 'Z.csv'
qoi_csv = 'CantileverBeam_QoIs.csv'
crystal_partition_csv = 'CantileverBeam_CrystalPartitions_maxStress.csv'

# Get input parameters, QoIs, and crystal partitions from file
latent_df = pd.read_csv(latent_csv, header=None)
qoi_df = pd.read_csv(base_path + qoi_csv)
crystal_df = pd.read_csv(base_path + crystal_partition_csv, header=None)

# Get Samples belonging to specified crystal
crystal = crystal_df.loc[persistence_level, crystal_df.loc[persistence_level, :] == crystal_id].index.to_numpy()

# Get QoI (x) and Latent Space (y) for persistence level and single crystal
qoi_df_crystal = qoi_df.loc[crystal, :]
latent_df_crystal = latent_df.loc[crystal, :]

# Create general training and test data sets
mask = np.random.rand(len(qoi_df_crystal)) < 0.7
qoi_train_df = qoi_df_crystal[mask]
qoi_test_df = qoi_df_crystal[~mask]

latent_train_df = latent_df_crystal[mask]
latent_test_df = latent_df_crystal[~mask]

# Select single qoi of interest and create numpy arrays
train_x = qoi_train_df.loc[:, qoi_name].to_numpy().reshape((-1, 1))
train_y = latent_train_df.to_numpy()

test_x = qoi_test_df.loc[:, qoi_name].to_numpy().reshape((-1, 1))
test_y = latent_test_df.to_numpy()

# Predict value for test data and compute MSE
test_predicted = predict(test_x, train_x, train_y, sigma)
mse = mean_squared_error(test_y, test_predicted)
print('The MSE is: ' + str(mse))

# Generate 50 samples between qoi min and max
min_qoi = qoi_df_crystal.loc[:, qoi_name].min()
max_qoi = qoi_df_crystal.loc[:, qoi_name].max()
steps_size = (max_qoi - min_qoi) / number_of_steps
new_samples = np.arange(min_qoi, max_qoi, steps_size).reshape((-1, 1))

# predict latent space for new samples
new_predicted = predict(new_samples, np.concatenate((train_x, test_x)), np.concatenate((train_y, test_y)), sigma)
np.savetxt(generated_data_path + 'new_latent_space_for_PL' + str(persistence_level) + '_CID' + str(crystal_id) + '.csv',
           new_predicted, delimiter=',')
