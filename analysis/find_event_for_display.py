import os
import sys
import numpy as np
import h5py

PLOT = True

weights_path = sys.argv[1]
out_dir = sys.argv[2]
data_file = sys.argv[3]
try:
    eventid = int(sys.argv[4])
except IndexError:
    eventid = -1

try:
    os.makedirs(out_dir)
except OSError:
    pass

if PLOT:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {'red': ((0., 0., 0.), (1., 1., 1.)), 'green': ((0., 0., 0.), (1., 0., 0.)), 'blue': ((0., 1., 1.), (1., 0., 0.))}
    cmap = mpl.colors.LinearSegmentedColormap('rb', colors)

fname = os.path.basename(data_file)
fname = fname[:fname.rfind('.')]

with h5py.File(data_file, 'r') as source:
    x = source['cluster'][:]
    n = source['size'][:]
    yene = source['truth_energy'][:]
    yid = source['truth_pid'][:]
    raw = source['raw'][:]
    coordinates = source['coordinates'][:]

if eventid == -1:
    from models.combined import make_model

    model = make_model()
    model.load_weights(weights_path)
    
    eventids = np.arange(x.shape[0])
    
    energy_in_range = np.asarray((yene > 49.) & (yene < 51.)).nonzero()
    x = x[energy_in_range]
    n = n[energy_in_range]
    yene = yene[energy_in_range]
    yid = yid[energy_in_range]
    raw = raw[energy_in_range]
    eventids = eventids[energy_in_range]
    
    total_pu = np.sum(raw[:, :, 0] * (1. - raw[:, :, 1]), axis=1)
    pu_in_range = np.asarray((total_pu > 55000.) & (total_pu < 65000.)).nonzero()
    x = x[pu_in_range]
    n = n[pu_in_range]
    yene = yene[pu_in_range]
    yid = yid[pu_in_range]
    raw = raw[pu_in_range]
    eventids = eventids[pu_in_range]
    
    x_input = np.stack((x[:, :, 0] / 18., x[:, :, 1] / 18., x[:, :, 2] / 200., x[:, :, 3] * 0.5), axis=-1)
    
    prediction = model.predict([x_input, n], verbose=1)
    
    pred_y = np.squeeze(prediction[0])
    pred_yid = np.squeeze(prediction[1])
    
    relediff = np.abs(pred_y * 100. - yene) / yene
    crossentropy = -(yid * np.log(np.where(pred_yid > 0., pred_yid, 1.e-10)) + (1. - yid) * np.log(np.where(pred_yid < 1., 1. - pred_yid, 1.e-10)))
    
    good_prediction = np.asarray((relediff < 0.15) & (crossentropy < 0.1)).nonzero()
    yene = yene[good_prediction]
    yid = yid[good_prediction]
    raw = raw[good_prediction]
    eventids = eventids[good_prediction]

else:
    yene = yene[eventid:eventid + 1]
    yid = yid[eventid:eventid + 1]
    raw = raw[eventid:eventid + 1]
    eventids = np.array([eventid])

cluster_radius_sq = 6.4 ** 2
threshold = 120.

for iev in range(raw.shape[0]):
    yene_ev = yene[iev]
    yid_ev = yid[iev]
    raw_ev = raw[iev, :, 0]
    frac_ev = raw[iev, :, 1]
    eid_ev = eventids[iev]

    iseed = np.argmax(raw_ev)

    seed_axis = np.tile(coordinates[iseed, :2], (coordinates.shape[0], 1))
    
    dr2 = np.sum(np.square(coordinates[:, :2] - seed_axis), axis=1)
    in_radius = np.asarray((dr2 < cluster_radius_sq) & (raw_ev > threshold)).nonzero()
    out_of_radius = np.asarray((dr2 > cluster_radius_sq) & (raw_ev > threshold)).nonzero()

    size_scaling = np.sqrt(raw_ev / 40.)

    x_in = coordinates[in_radius, 2]
    y_in = coordinates[in_radius, 1]
    z_in = coordinates[in_radius, 0]
    c_in = frac_ev[in_radius]
    s_in = size_scaling[in_radius]

    x_out = coordinates[out_of_radius, 2]
    y_out = coordinates[out_of_radius, 1]
    z_out = coordinates[out_of_radius, 0]
    c_out = frac_ev[out_of_radius]
    s_out = size_scaling[out_of_radius]

    if yid_ev == 0:
        part = 'electron'
    else:
        part = 'pion'

    if PLOT:
        ax.clear()
       
        ax.scatter(xs=x_in, ys=y_in, zs=z_in, c=c_in, s=s_in, alpha=1, cmap=cmap)
        ax.scatter(xs=x_out, ys=y_out, zs=z_out, c=c_out, s=s_out, alpha=0.3, cmap=cmap)
   
        fig.savefig('%s/%s_%s_%d.pdf' % (out_dir, part, fname, eid_ev))
        fig.savefig('%s/%s_%s_%d.png' % (out_dir, part, fname, eid_ev))

    else:
        yene_in = np.sum((raw_ev * frac_ev)[in_radius]) * 1.e-3
        pu_ev = np.sum(raw_ev * (1. - frac_ev)) * 1.e-3
        pu_in = np.sum((raw_ev * (1. - frac_ev))[in_radius]) * 1.e-3

        print('%s Eprim = %f (%f) PU = %f (%f)' % (part, yene_ev, yene_in, pu_ev, pu_in))

        in_size = s_in.shape[0]
        out_size = s_out.shape[0]
        with h5py.File('%s/%s_%s_%d.h5' % (out_dir, part, fname, eid_ev), 'w', libver='latest') as out_file:
            out_file.create_dataset('prim', (2,), dtype='f').write_direct(np.array([yene_ev, yene_in]))
            out_file.create_dataset('pu', (2,), dtype='f').write_direct(np.array([pu_ev, pu_in]))
            out_file.create_dataset('pid', (1,), dtype='i').write_direct(np.array([yid_ev], dtype=np.int8))
            out_file.create_dataset('x_in', (in_size,), compression='gzip', dtype='f').write_direct(x_in)
            out_file.create_dataset('y_in', (in_size,), compression='gzip', dtype='f').write_direct(y_in)
            out_file.create_dataset('z_in', (in_size,), compression='gzip', dtype='f').write_direct(z_in)
            out_file.create_dataset('c_in', (in_size,), compression='gzip', dtype='f').write_direct(c_in)
            out_file.create_dataset('s_in', (in_size,), compression='gzip', dtype='f').write_direct(s_in)
            out_file.create_dataset('x_out', (out_size,), compression='gzip', dtype='f').write_direct(x_out)
            out_file.create_dataset('y_out', (out_size,), compression='gzip', dtype='f').write_direct(y_out)
            out_file.create_dataset('z_out', (out_size,), compression='gzip', dtype='f').write_direct(z_out)
            out_file.create_dataset('c_out', (out_size,), compression='gzip', dtype='f').write_direct(c_out)
            out_file.create_dataset('s_out', (out_size,), compression='gzip', dtype='f').write_direct(s_out)
