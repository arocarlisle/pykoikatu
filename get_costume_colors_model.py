import os

import h5py
import numpy as np

from get_body_params_model import get_gmm

in_dir = 'data_booru'
out_dir = 'data'

if __name__ == '__main__':
    with h5py.File(os.path.join(in_dir, 'costume_colors.hdf5'), 'r') as f:
        costume_colors = np.array(f['data'], dtype=np.float64)

    costume_colors = costume_colors.reshape([-1, costume_colors.shape[-1]])
    gmm = get_gmm(costume_colors, 18)

    with h5py.File(os.path.join(out_dir, 'costume_colors.hdf5'), 'w') as f:
        f.create_dataset('weight', data=gmm.weights_)
        f.create_dataset('mean', data=gmm.means_)
        f.create_dataset('cov', data=gmm.covariances_)
