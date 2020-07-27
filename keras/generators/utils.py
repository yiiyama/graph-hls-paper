import numpy as np

def make_dataset(gen, n_steps, truth_only=False):
    item = gen[0]
    item_0_list = type(item[0]) is list

    if n_steps is None:
        n_steps = len(gen)
        if item_0_list:
            getitem = lambda ibatch: gen[ibatch]
        else:
            def getitem(ibatch):
                item = gen[ibatch]
                return ([item[0]], item[1])
    else:
        if item_0_list:
            getitem = lambda _: next(gen)
        else:
            def getitem(_):
                item = next(gen)
                return ([item[0]], item[1])

    if item_0_list:
        ntotal = sum(item[0][-1].shape[0] for ibatch in range(n_steps))
    else:
        ntotal = sum(item[0].shape[0] for ibatch in range(n_steps))

    x = None
    truth = None
    start = 0
    for ibatch in range(n_steps):
        item = getitem(ibatch)
        end = start + item[0][-1].shape[0]

        if not truth_only:
            if x is None:
                x = [np.empty((ntotal,) + v.shape[1:], dtype=v.dtype) for v in item[0]]
    
            for ix in range(len(x)):
                x[ix][start:end] = item[0][ix]

        if truth is None:
            if type(item[1]) is dict:
                truth = dict((key, np.empty((ntotal,) + value.shape[1:])) for key, value in item[1].items())
            elif type(item[1]) is list:
                truth = [np.empty((ntotal,) + value.shape[1:]) for value in item[1]]
            else:
                truth = np.empty((ntotal,) + item[1].shape[1:])

        if type(item[1]) is dict:
            for key, value in item[1].items():
                truth[key][start:end] = value
        elif type(item[1]) is list:
            for i, value in enumerate(item[1]):
                truth[i][start:end] = value
        else:
            truth[start:end] = item[1]

        start = end

    if truth_only:
        return truth
    else:
        return x, truth

def to_dense(n, x, n_vert_max=None, x_dense=None):
    if x_dense is None:
        x_dense = np.zeros((n.shape[0], n_vert_max, x.shape[1]), dtype=np.float)

    batch_indices = np.repeat(np.arange(n.shape[0]), n)
    cluster_indices = np.r_[tuple(np.s_[:i] for i in n)]
    x_dense[batch_indices, cluster_indices] = x

    return x_dense
