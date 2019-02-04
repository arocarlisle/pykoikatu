import os

import h5py
import numpy as np

in_dir = 'data_8300'
out_dir = 'data'

if __name__ == '__main__':
    with h5py.File(os.path.join(in_dir, 'body_params.hdf5'), 'r') as f:
        body_params = np.array(f['data'], dtype=np.float64)

    body_params = body_params.T
    mean = np.mean(body_params, axis=1)
    cov = np.cov(body_params)

    with h5py.File(os.path.join(out_dir, 'body_params.hdf5'), 'w') as f:
        f.create_dataset('mean', data=mean)
        f.create_dataset('cov', data=cov)
