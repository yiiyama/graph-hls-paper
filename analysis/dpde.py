import os
import sys
import numpy as np
import h5py

PLOT = False

from models.combined import make_model

weights_path = sys.argv[1]
out_name = sys.argv[2]
data_file = sys.argv[3]
event_id = int(sys.argv[4])

try:
    os.makedirs(os.path.dirname(out_name))
except OSError:
    pass

model = make_model()
model.load_weights(weights_path)

rdelta = 0.1
cluster_radius_sq = 6.4 ** 2
threshold = 120. # hit energy threshold

with h5py.File(data_file, 'r') as source:
    x = source['cluster'][event_id]
    n = source['size'][event_id]
    raw_ev = source['raw'][event_id, :, 0]
    coordinates = source['coordinates'][:]

iseed = np.argmax(raw_ev)

x_input = np.tile(np.expand_dims(x, axis=0), (n + 1, 1, 1))
indices = (np.arange(n), np.arange(n), 3)
x_input[indices] *= (1. + rdelta)
x_input = np.stack((x_input[:, :, 0] / 18., x_input[:, :, 1] / 18., x_input[:, :, 2] / 200., x_input[:, :, 3] * 0.5), axis=-1)

prediction = model.predict([x_input, np.array([n] * (n + 1))], verbose=1)
pred_y = np.squeeze(prediction[0])

dp = (pred_y[:-1] - pred_y[-1]) * 1.e+2
de = x[:n, 3] * rdelta

x_in = np.array(x[:n, 2])
y_in = x[:n, 1] + coordinates[iseed, 1]
z_in = x[:n, 0] + coordinates[iseed, 0]
c_in = dp / de
s_in = np.sqrt(x[:n, 3] * 1.e+3 / 40.)

seed_axis = np.tile(coordinates[iseed, :2], (coordinates.shape[0], 1))

dr2 = np.sum(np.square(coordinates[:, :2] - seed_axis), axis=1)
out_of_radius = np.asarray((dr2 > cluster_radius_sq) & (raw_ev > threshold)).nonzero()

x_out = coordinates[out_of_radius, 2]
y_out = coordinates[out_of_radius, 1]
z_out = coordinates[out_of_radius, 0]
s_out = np.sqrt(raw_ev[out_of_radius] / 40.)

if PLOT:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #cmap_in = get_cmap('cool').reversed()
    cmap_in = get_cmap('winter')
    c_out = np.ones_like(s_out) * 0.5
    cmap_out = get_cmap('Greys')
    
    ax.scatter(xs=x_in, ys=y_in, zs=z_in, c=c_in, s=s_in, alpha=1, vmin=0., vmax=1.4, cmap=cmap_in)
    ax.scatter(xs=x_out, ys=y_out, zs=z_out, c=c_out, s=s_out, alpha=0.3, vmin=0., vmax=1., cmap=cmap_out)

    fig.savefig('%s.pdf' % out_name)
    fig.savefig('%s.png' % out_name)

else:
    in_size = s_in.shape[0]
    out_size = s_out.shape[0]
    with h5py.File('%s.h5' % out_name, 'w', libver='latest') as out_file:
        out_file.create_dataset('x_in', (in_size,), compression='gzip', dtype='f').write_direct(x_in)
        out_file.create_dataset('y_in', (in_size,), compression='gzip', dtype='f').write_direct(y_in)
        out_file.create_dataset('z_in', (in_size,), compression='gzip', dtype='f').write_direct(z_in)
        out_file.create_dataset('c_in', (in_size,), compression='gzip', dtype='f').write_direct(c_in)
        out_file.create_dataset('s_in', (in_size,), compression='gzip', dtype='f').write_direct(s_in)
        out_file.create_dataset('x_out', (out_size,), compression='gzip', dtype='f').write_direct(x_out)
        out_file.create_dataset('y_out', (out_size,), compression='gzip', dtype='f').write_direct(y_out)
        out_file.create_dataset('z_out', (out_size,), compression='gzip', dtype='f').write_direct(z_out)
        out_file.create_dataset('s_out', (out_size,), compression='gzip', dtype='f').write_direct(s_out)
