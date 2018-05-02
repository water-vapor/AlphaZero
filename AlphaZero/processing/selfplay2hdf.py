import h5py as h5
import argparse
import numpy as np
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert selfplay data from npz to hdf5.')
    parser.add_argument('dir', type=str, help='dir of npz files')
    parser.add_argument('out', type=str, help='output dir of hdf5 file')
    args = parser.parse_args()

    size = 0
    count = 0

    with h5.File(args.out, 'w') as h5f:

        states = h5f.require_dataset(
            'states',
            dtype=np.uint8,
            shape=(1, 17, 19, 19),
            maxshape=(None, 17, 19, 19),  # 'None' == arbitrary size
            exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
            chunks=(64, 17, 19, 19),  # approximately 1MB chunks
            compression="lzf")
        actions = h5f.require_dataset(
            'actions',
            dtype=np.uint8,
            shape=(1, 2),
            maxshape=(None, 2),
            exact=False,
            chunks=(1024, 2),
            compression="lzf")
        results = h5f.require_dataset(
            'results',
            dtype=np.int8,
            shape=(1, 1),
            maxshape=(None, 1),
            exact=False,
            chunks=(1024, 1),
            compression="lzf")

        for file in os.listdir(args.dir):
            if file.endswith('.npz'):
                loaded = np.load(os.path.join(args.dir, file))
                max_pos_lin = np.argmax(loaded['arr_1'], axis=1)
                max_pos_h = max_pos_lin // 19
                max_pos_w = max_pos_lin % 19
                max_pos = np.stack([max_pos_h, max_pos_w], 1)
                for i in range(loaded['arr_2'].shape[0]):
                    if loaded['arr_2'][i] == 0:
                        loaded['arr_2'][i] = -1
                res = np.expand_dims(loaded['arr_2'], 1)
                data = (loaded['arr_0'], max_pos, loaded['arr_2'])
                print(loaded['arr_0'].shape, loaded['arr_1'].shape, loaded['arr_2'].shape)
                states.resize((size+loaded['arr_0'].shape[0], 17, 19, 19))
                states[size:size+loaded['arr_0'].shape[0]] = loaded['arr_0']
                actions.resize((size + loaded['arr_0'].shape[0], 2))
                actions[size:size + loaded['arr_0'].shape[0]] = max_pos
                results.resize((size + loaded['arr_0'].shape[0], 1))
                results[size:size + loaded['arr_0'].shape[0]] = res
                size += loaded['arr_0'].shape[0]
                print('{} data size: {}'.format(count, size))
                count += 1
