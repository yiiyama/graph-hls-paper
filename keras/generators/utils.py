import numpy as np

def to_dense(n, x, n_vert_max=None, x_dense=None, features=None):
    if x_dense is None:
        x_dense = np.zeros((n.shape[0], n_vert_max, x.shape[1]), dtype=np.float)

    batch_indices = np.repeat(np.arange(n.shape[0]), n)
    cluster_indices = np.r_[tuple(np.s_[:i] for i in n)]

    if features is None:
        x_dense[batch_indices, cluster_indices] = x
    else:
        x_dense[batch_indices, cluster_indices] = x[:, features]

    return x_dense
