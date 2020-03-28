import random
import h5py

def make_generator(paths, batch_size, num_steps=None, format='xn', features=None, n_vert_max=256, y_dtype=np.int, y_features=None, sample_weighting=None, dataset_name='events'):
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
                        if features is None:
                            x = f['x'][start:end]
                        else:
                            x = f['x'][start:end][, :, features]

                        if y_features is None:
                            y = f['y'][start:end]
                        else:
                            y = f['y'][start:end][:, y_features]

                        yield x, f['n'][start:end]], y

                        start = end
                        end += batch_size

    return get_event(), n_steps
