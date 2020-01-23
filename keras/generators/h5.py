import random
import h5py

def make_generator(paths, batch_size, features=None, n_vert_max=256, y_shape=None, dataset_name=''):
    paths = list(paths)

    n_steps = 0
    for path in paths:
        with h5py.File(path, 'r') as f:
            n_steps += f['n'].shape[0] // batch_size

    def get_event():
        while True:
            random.shuffle(paths)

            for path in paths:
                with h5py.File(path, 'r') as f:
                    start = 0
                    end = batch_size
                    while start < f['n'].shape[0] - 1:
                        yield [f['x'][start:end, :, features], f['n'][start:end]], f['y'][start:end]

                        start = end
                        end += batch_size

    return get_event(), n_steps
