from __future__ import absolute_import, division, print_function, unicode_literals

def make_generator(paths, batch_size, features=None, n_vert_max=256, y_shape=None, dataset_name='events'):
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
