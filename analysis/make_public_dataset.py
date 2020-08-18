import sys
import glob
import uproot
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d # necessary to use projection='3d'

source = '/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/generated2'
plotdir = '/afs/cern.ch/user/y/yiiyama/www/plots/200410_event_display'

cluster_radius_sq = 6.4 ** 2
threshold = 120. # hit energy threshold

geom_data = uproot.open('../generation/geom.root')['detector'].arrays(['x', 'y', 'z'], namedecode='ascii')
coords = np.stack((geom_data['x'], geom_data['y'], geom_data['z']), axis=-1)

processed_paths_list = []
def make_generator(paths, branches, report_ievt=False):
    def get_event():
        current_path = ''
        for path, data in uproot.iterate(paths, 'events', branches, namedecode='ascii', reportpath=True):
            if path != current_path:
                processed_paths_list.append(path)

            for ievt in range(data[branches[0]].shape[0]):
                yield tuple(data[b][ievt] for b in branches)

    return get_event

ele_paths = glob.glob('%s/electron_10_100/*/events_*.root' % source)
pi_paths = glob.glob('%s/pioncharged_10_100/*/events_*.root' % source)
pu_paths = glob.glob('%s/pileup_0_0/*/events_*.root' % source)

prims = make_generator(prim_paths, ['recoEnergy', 'genEnergy'])()
pus = make_generator(pu_paths, ['recoEnergy'])()

for iev, ipart in enumerate(np.random.randint(0, 2, args.nevt)):
    event = make_event(ipart)


import glob
import h5py
import numpy as np
from generators.uproot_jagged_keep import make_generator
from generators.utils import make_dataset

in_path = glob.glob('/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/mixed/combined_new/events_*.root')
out_path = '/tmp/yiiyama/toy_calorimeter_clusters.h5'
n_vert_max = 128
batch_size = 128
n_batches = 512

gen, n_steps = make_generator(in_path, batch_size, n_batches, dataset_name='events', n_vert_max=n_vert_max, y_features=[0, 1], y_dtype=[np.float, np.float])

x, y = make_dataset(gen, n_steps)

n_events = batch_size * n_batches
n_feat = 4

with h5py.File(out_path, 'w', libver='latest') as output:
    chunk_size = (batch_size, n_vert_max, n_feat)
    clusters = output.create_dataset('vertices', (n_events, n_vert_max, n_feat), chunks=chunk_size, compression='gzip', dtype='f')
    clusters.write_direct(x[0])
    sizes = output.create_dataset('num', (n_events,), chunks=(batch_size,), compression='gzip', dtype='i')
    sizes.write_direct(x[1])
    pid = output.create_dataset('pid', (n_events,), chunks=(batch_size,), compression='gzip', dtype='i')
    pid[:] = y[0]
    energy = output.create_dataset('energy', (n_events,), chunks=(batch_size,), compression='gzip', dtype='f')
    energy.write_direct(y[1])
