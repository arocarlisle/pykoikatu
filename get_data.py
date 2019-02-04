import os

from pykoikatu import *

in_dir = 'test_booru'
out_dir = 'data_booru'

if __name__ == '__main__':
    body_params_data = []
    hair_ids_data = []
    costume_ids_data = []
    costume_colors_data = []
    for root, dirs, files in os.walk(in_dir):
        for file in sorted(files):
            in_filename = os.path.join(root, file)
            print(in_filename)
            card = read_card(in_filename)
            body_params_data.append(parse_body_params(card))
            hair_ids_data.append(parse_hair_ids(card))
            costume_ids_data.append(parse_costume_ids(card))
            costume_colors_data.append(parse_costume_colors(card))

    with h5py.File(os.path.join(out_dir, 'body_params.hdf5'), 'w') as f:
        f.create_dataset('data', data=body_params_data)
    with h5py.File(os.path.join(out_dir, 'hair_ids.hdf5'), 'w') as f:
        f.create_dataset('data', data=hair_ids_data)
    with h5py.File(os.path.join(out_dir, 'costume_ids.hdf5'), 'w') as f:
        f.create_dataset('data', data=costume_ids_data)
    with h5py.File(os.path.join(out_dir, 'costume_colors.hdf5'), 'w') as f:
        f.create_dataset('data', data=costume_colors_data)
