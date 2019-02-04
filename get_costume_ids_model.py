import os

import h5py
import numpy as np

in_dir = 'data_booru'
out_dir = 'data'
part_names = [
    'top', 'bot', 'bra', 'shorts', 'gloves', 'panst', 'socks', 'shoes', 'shoes'
]


def read_item_ids(part_name):
    item_ids = []
    with open(
            'item_lists/co_{}.txt'.format(part_name), 'r',
            encoding='utf-8') as f:
        for row in f:
            item_ids.append(int(row.split()[0]))
    return item_ids


if __name__ == '__main__':
    with h5py.File(os.path.join(in_dir, 'costume_ids.hdf5'), 'r') as f:
        costume_ids = np.array(f['data'], dtype=np.int64)
    costume_ids = costume_ids.reshape([-1, costume_ids.shape[-1]])

    parts_ids = {
        part_name: read_item_ids(part_name)
        for part_name in set(part_names)
    }
    max_parts_id = [max(parts_ids[part_name]) + 1 for part_name in part_names]
    costume_ids = costume_ids[(costume_ids < np.array(max_parts_id)).all(
        axis=1), :]
