import os

import h5py
import numpy as np

in_dir = 'data_booru'
out_dir = 'data'

if __name__ == '__main__':
    with h5py.File(os.path.join(in_dir, 'costume_colors.hdf5'), 'r') as f:
        costume_colors = np.array(f['data'], dtype=np.float64)

    costume_colors = costume_colors.reshape([-1, costume_colors.shape[-1]]).T
    mean = np.mean(costume_colors, axis=1)
    cov = np.cov(costume_colors)

    with h5py.File(os.path.join(out_dir, 'costume_colors.hdf5'), 'w') as f:
        f.create_dataset('mean', data=mean)
        f.create_dataset('cov', data=cov)
