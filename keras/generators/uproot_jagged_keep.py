from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import uproot

max_cluster_size = 256

def make_generator(paths, batch_size, features=None):
    n_events = 0
    for data in uproot.iterate(paths, 'events', ['n'], namedecode='ascii'):
        n_events += data['n'].shape[0]

    n_steps = n_events // batch_size

    n = np.empty((n_events,), dtype=np.int)
    y = np.empty((n_events,), dtype=np.int)

    start = 0
    for data in uproot.iterate(paths, 'events', ['n', 'y'], namedecode='ascii'):
        end = start + data['n'].shape[0]
        n[start:end] = data['n']
        y[start:end] = data['y']
        start = end

    if features is None:
        for data in uproot.iterate(paths, 'events', ['x'], namedecode='ascii', entrysteps=1):
            nfeat = data['x'].content.shape[1]
            break
    else:
        nfeat = len(features)

    nhits_total = np.sum(n)
    xcont = np.empty((nhits_total, nfeat), dtype=np.float)

    cstart = 0
    istart = 0
    for data in uproot.iterate(paths, 'events', ['x'], namedecode='ascii'):
        x = data['x'].content
        cend = cstart + x.shape[0]
        if features is None:
            xcont[cstart:cend] = x
        else:
            xcont[cstart:cend] = x[:, features]

        cstart = cend

    def get_event():
        v_x = np.zeros((batch_size, max_cluster_size, nfeat), dtype=np.float)

        while True:
            xpos = 0
            for i_step in range(n_steps):
                start = batch_size * i_step
                end = start + batch_size

                batch_indices = np.repeat(np.arange(batch_size), n[start:end])
                cluster_indices = np.r_[tuple(np.s_[:i] for i in n[start:end])]

                v_x *= 0.

                if features is None:
                    v_x[batch_indices, cluster_indices] = xcont[xpos:xpos + batch_indices.shape[0]]
                else:
                    v_x[batch_indices, cluster_indices] = x[xpos:xpos + batch_indices.shape[0], features]

                xpos += batch_indices.shape[0]

                yield [v_x, n[start:end]], y[start:end]

    return get_event, n_steps


if __name__ == '__main__':
    generator, n_steps = make_generator('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/mixed/classification_nopu/root-sparse/events_0.root', 2)

    print(n_steps, 'steps')
    print(next(generator()))
