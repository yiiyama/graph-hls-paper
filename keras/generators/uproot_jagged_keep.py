from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import uproot

from .utils import to_dense

def make_generator(paths, batch_size, features=None, n_vert_max=256, y_shape=None, y_dtype=np.int, y_features=None, dataset_name='events'):
    n_events = 0
    for data in uproot.iterate(paths, dataset_name, ['n'], namedecode='ascii'):
        n_events += data['n'].shape[0]

    n_steps = n_events // batch_size

    n = np.empty((n_events,), dtype=np.int)
    if y_shape is None:
        shape = (n_events,)
    elif type(y_shape) is tuple:
        shape = (n_events,) + y_shape
    else:
        shape = (n_events, y_shape)
    y = np.empty(shape, dtype=y_dtype)

    start = 0
    for data in uproot.iterate(paths, dataset_name, ['n', 'y'], namedecode='ascii'):
        end = start + data['n'].shape[0]
        n[start:end] = data['n']
        if type(y_shape) is not None: # figure out how to do this for tuple y_shape
            y[start:end] = data['y'][:, y_features]
        else:
            y[start:end] = data['y']
        start = end

    if features is None:
        for data in uproot.iterate(paths, dataset_name, ['x'], namedecode='ascii', entrysteps=1):
            nfeat = data['x'].content.shape[1]
            break
    else:
        nfeat = len(features)

    nhits_total = np.sum(n)
    xcont = np.empty((nhits_total, nfeat), dtype=np.float)

    cstart = 0
    istart = 0
    for data in uproot.iterate(paths, dataset_name, ['x'], namedecode='ascii'):
        x = data['x'].content
        cend = cstart + x.shape[0]
        if features is None:
            xcont[cstart:cend] = x
        else:
            xcont[cstart:cend] = x[:, features]

        cstart = cend

    def get_event():
        v_x = np.zeros((batch_size, n_vert_max, nfeat), dtype=np.float)

        while True:
            xpos = 0
            for i_step in range(n_steps):
                start = batch_size * i_step
                end = start + batch_size
                n_x = np.sum(n[start:end])

                v_x *= 0.
                to_dense(n[start:end], xcont[xpos:xpos + n_x], x_dense=v_x)

                xpos += n_x

                yield [v_x, n[start:end]], y[start:end]

    return get_event, n_steps


if __name__ == '__main__':
    import sys

    path = sys.argv[1]

    generator, n_steps = make_generator(path, 2, n_vert_max=1024, y_shape=1, y_features=[0])

    print(n_steps, 'steps')
    print(next(generator()))
