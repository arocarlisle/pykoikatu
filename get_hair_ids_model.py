import os

import h5py
import numpy as np

from pykoikatu import hair_part_names, read_max_parts_id
from trdm import TensorRankDecompositionModel

in_dir = 'data_8300'
out_dir = 'data'


def n_parameters(model):
    return model.k - 1 + sum([model.k * (d - 1) for d in model.ds])


def bic(model, data):
    return (-2 * model.score(data) * data.shape[0] +
            n_parameters(model) * np.log(data.shape[0]))


# Choose n_components to minimize bic
def get_trdm(data, shape, n_components):
    trdm = TensorRankDecompositionModel(shape, n_components, verbose=2)
    trdm.init_km(data)
    trdm.fit(data)
    print('weight')
    print(trdm.w)
    print('BIC', bic(trdm, data))
    return trdm


if __name__ == '__main__':
    with h5py.File(os.path.join(in_dir, 'hair_ids.hdf5'), 'r') as f:
        hair_ids = np.array(f['data'], dtype=np.int64)

    max_parts_id = read_max_parts_id('bo', hair_part_names)
    hair_ids = hair_ids[(hair_ids < np.array(max_parts_id)).all(axis=1), :]
    print(hair_ids.shape[0])

    trdm = get_trdm(hair_ids, max_parts_id, 50)

    with h5py.File(os.path.join(out_dir, 'hair_ids.hdf5'), 'w') as f:
        f.create_dataset('weight', data=trdm.w)
        for i, u in enumerate(trdm.us):
            f.create_dataset('u{}'.format(i), data=u)
