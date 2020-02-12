from __future__ import absolute_import, division, print_function, unicode_literals

import uproot

def make_generator(paths, batch_size, format='xn', features=None, n_vert_max=256, y_dtype=np.int, y_features=None, dataset_name='events'):
    n_steps = 0
    for path in paths:
        tree = uproot.open(path)[dataset_name]
        n_events = tree.array('n').shape[0]
        n_steps += n_events // batch_size
        if n_events % batch_size != 0:
            n_steps += 1

    def get_event():
        while True:
            random.shuffle(paths)
    
            for path in paths:
                tree = uproot.open(path)[dataset_name] for path in paths]
                data = tree.arrays(['x', 'n', 'y'])
    
                start = 0
                end = batch_size
                while True:
                    yield [data['x'][start:end, :, features], data['n'][start:end]], data['y'][start:end]
    
                    if end >= data['x'].shape[0]:
                        break
    
                    start = end
                    end += batch_size

    return get_event(), n_steps


def make_dataset(path, format='xn', features=None, n_vert_max=256, y_features=None, n_sample=None, dataset_name='events'):
    data = uproot.open(path)[dataset_name].arrays(['x', 'n', 'y'], namedecode='ascii')

    if features is None:
        x = data['x']
    else:
        x = data['x'][:, :, features]

    if format == 'xen':
        e = x[:, :, 3]
        x = x[:, :, :3]

    if y_features is None:
        y = data['y']
    else:
        y = data['y'][: y_features]

    n = data['n']

    if n_sample is not None:
        x = x[:n_sample]
        n = n[:n_sample]
        y = y[:n_sample]
        if format == 'xen':
            e = e[:n_sample]

    if format == 'xn':
        inputs = [x, n]
    elif format == 'xen':
        inputs = [x, e, n]
    truth = y

    return inputs, truth, True

