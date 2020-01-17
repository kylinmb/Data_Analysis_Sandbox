"""
This example uses the kernel regression model to generate new
latent space values from QoIs
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import sys
sys.path.append("../../..")
from regression.kernel_regression.models.gaussian_kernel import predict, predict_matrix

# So results are reproducible
np.random.seed(101)

# Settings
persistence_level = 10
crystal_id = 9
qoi_name = 'Max Stress'
sigma = 0.25
number_of_steps = 1 #50

# Path Settings
generated_data_path = '../generated_data/'
#base_path = '/usr/sci/projects/dSpaceX/DATA/CantileverBeam_wclust_wraw/'
base_path = '/Users/cam/data/dSpaceX/DATA/CantileverBeam_wclust_wraw/'
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
all_qoi_data = qoi_df_crystal.loc[:, qoi_name].to_numpy().reshape((-1, 1))
all_latent_data = latent_df_crystal.to_numpy()

# Create general training and test data sets
# mask = np.random.rand(len(qoi_df_crystal)) < 0.7
# qoi_train_df = qoi_df_crystal[mask]
# qoi_test_df = qoi_df_crystal[~mask]

# latent_train_df = latent_df_crystal[mask]
# latent_test_df = latent_df_crystal[~mask]

# Select single qoi of interest and create numpy arrays
# train_x = qoi_train_df.loc[:, qoi_name].to_numpy().reshape((-1, 1)) # 1d QoIs of the samples we have 
# train_y = latent_train_df.to_numpy() # the matrix of all input samples' latent space values

# test_x = qoi_test_df.loc[:, qoi_name].to_numpy().reshape((-1, 1)) # just the N new QoIs along the crystal between [cmin, cmax]
# test_y = latent_test_df.to_numpy() # we won't have test y; this was just to verify new z_coord generation worked by using the z_coords from other sample in the dataset - called "hold out cross validation"; could do "hold one out" in order to use the minimum amount of hold out data and then cycle through to ensure the z_coords being generated are the same (or very close) to those created by the model

# Predict value for test data and compute MSE (mean squared error, average of all the squared errors just used to see how close these 
#test_predicted = predict(test_x, train_x, train_y, sigma)
#matrix_prediction = predict_matrix(test_x, train_x, train_y, sigma)
#mse_for = mean_squared_error(test_y, test_predicted)            # this was just to verify the for loop and matrix forms of the computation were the same
#mse_matrix = mean_squared_error(test_y, matrix_prediction)
#print('The MSE using the for loop is: ' + str(mse_for))
#print('The MSE matrix using the for loop is: ' + str(mse_matrix))

# Generate 50 samples between qoi min and max
min_qoi = qoi_df_crystal.loc[:, qoi_name].min()
max_qoi = qoi_df_crystal.loc[:, qoi_name].max()
steps_size = (max_qoi - min_qoi) / number_of_steps
new_samples = np.arange(min_qoi, max_qoi, steps_size).reshape((-1, 1)) # QoIs, just like to 

# predict latent space for new samples
# <ctc> this will be more like my call, using all the data: I won't even be concatenating; sigma is the gaussian stddev, so it could be smaller larger
new_predicted = predict_matrix(new_samples, all_qoi_data, all_latent_data, sigma)                   # this is all we need, uses all the original z_coords (train and test) -> note that there is a new c++ function in a branch that is predict (the for-loops, etc), but it hasn't really been tested. So if I get a wild hair I can put this in to be called by the fetchN Controller function (that currently just called the fetchALL) 

# np.savetxt(generated_data_path + 'new_latent_space_for_PL' + str(persistence_level) + '_CID' + str(crystal_id) + '.csv',
#            new_predicted, delimiter=',')
