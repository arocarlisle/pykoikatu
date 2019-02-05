import os

import h5py
import numpy as np

from get_hair_ids_model import get_trdm
from pykoikatu import costume_part_names, read_max_parts_id

in_dir = 'data_8300'
out_dir = 'data'

if __name__ == '__main__':
    with h5py.File(os.path.join(in_dir, 'costume_ids.hdf5'), 'r') as f:
        costume_ids = np.array(f['data'], dtype=np.int64)
    costume_ids = costume_ids.reshape([-1, costume_ids.shape[-1]])

    max_parts_id = read_max_parts_id('co', costume_part_names)
    costume_ids = costume_ids[(costume_ids < np.array(max_parts_id)).all(
        axis=1), :]
    print(costume_ids.shape[0])

    trdm = get_trdm(costume_ids, max_parts_id, 50)

    with h5py.File(os.path.join(out_dir, 'costume_ids.hdf5'), 'w') as f:
        f.create_dataset('weight', data=trdm.w)
        for i, u in enumerate(trdm.us):
            f.create_dataset('u{}'.format(i), data=u)
