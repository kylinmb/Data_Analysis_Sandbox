import numpy as np
import pandas as pd

from regression.kernel_regression.models.gaussian_kernel import predict_matrix

np.random.seed(101)

qoi_name = 'Compliance'
label_id = '0'
number_of_steps = 50
generated_data_path = '../generated_data/' # path to store results
base_path = '/usr/sci/projects/dSpaceX/DATA/DataFromWei_01_24_2020/'
latent_csv = base_path + 'lebel='+label_id+'.csv'

# load latent space
latent_space = pd.read_csv(latent_csv, dtype={'laten1': np.float64, 'latent2': np.float64, 'image_id': np.int32})
latent_space['image_id'] -= 1 # zero index sample id

# load qoi and select rows belonging to crystal
qoi_csv = base_path + 'CantileverBeam_QoIs.csv'
qoi = pd.read_csv(qoi_csv)
qoi_for_crystal = qoi.iloc[latent_space['image_id']][qoi_name].to_numpy().reshape((-1, 1))

# drop image_id column
latent_space = latent_space.drop(labels='image_id', axis=1).to_numpy()

# Get new samples
min_qoi = qoi_for_crystal.min()
max_qoi = qoi_for_crystal.max()
step_size = (max_qoi - min_qoi) / (number_of_steps - 1)
new_samples = np.arange(min_qoi, max_qoi + step_size, step_size).reshape((-1, 1))
sigma = step_size * 0.15

predicted_latent_space = predict_matrix(new_samples, qoi_for_crystal, latent_space, sigma)
results = pd.DataFrame({'Latent1': predicted_latent_space[:, 0], 'Latent2': predicted_latent_space[:, 1], 'QoI': new_samples[:, 0]})
results.to_csv(path_or_buf=(generated_data_path + 'new_latent_space_qoi='+qoi_name.replace(" ", "")+'_label='+label_id+'.csv'), index=False)



