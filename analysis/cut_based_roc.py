import sys
import numpy as np
import uproot
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

FROM_ROOT = False

output_path = sys.argv[1]

if FROM_ROOT:
    from generators.uproot_jagged_keep import make_generator
    from generators.utils import make_dataset
    
    data_path = '/eos/cms/store/cmst3/user/yiiyama/graph_hls_paper/mixed/combined_new/events_4.root'
    
    gen, n_steps = make_generator([data_path], 128, num_steps=None, y_features=0, y_dtype=np.float)
    inputs, truth = make_dataset(gen, n_steps, truth_only=False)
    x, n = inputs
    
    with h5py.File('/tmp/yiiyama/events_4.h5', 'w') as out:
        out.create_dataset('x', x.shape).write_direct(x)
        out.create_dataset('n', n.shape).write_direct(n)
        out.create_dataset('truth', truth.shape).write_direct(truth)

else:
    with h5py.File('/tmp/yiiyama/events_4.h5', 'r') as source:
        x = source['x'][:]
        n = source['n'][:]
        truth = source['truth'][:]

# wrong geometry was used in mixing - true z max is 200 but was set to 120 in geometry file, mixing still normalized by 200
#ecal = np.sum(np.where(x[:, :, 2] < (25. * 120. / 200.) / 200., x[:, :, 3], 0.), axis=1)
#hcal = np.sum(np.where(x[:, :, 2] > (25. * 120. / 200.) / 200., x[:, :, 3], 0.), axis=1)
#h_over_e = np.ones_like(ecal)
#np.divide(hcal, ecal, out=h_over_e, where=(ecal != 0.))
etotal = np.sum(x[:, :, 3], axis=1)
zweighted = x[:, :, 2] * x[:, :, 3]
depth = np.sum(zweighted, axis=1) / etotal
zrms = np.sqrt(np.sum(np.square(x[:, :, 2]) * x[:, :, 3], axis=1) / etotal - np.square(depth))
#rweighted = np.sqrt(np.sum(np.square(x[:, :, 0:2]), axis=2)) * x[:, :, 3]
#width = np.sqrt(np.sum(np.square(rweighted), axis=1))
#width = np.sum(np.square(np.std(x[:, :, 0:2] * x[:, :, 3:], axis=1)), axis=1)

ele = np.asarray(truth == 0).nonzero()[0]
pi = np.asarray(truth == 1).nonzero()[0]

fig = plt.figure()


#fig.add_subplot(211)
##plt.hist([h_over_e[ele], h_over_e[pi]], bins=100, range=(0., 2.), label=['ele', 'pi'])
#plt.hist([zrms[ele], zrms[pi]], bins=100, range=(0., 0.2), label=['ele', 'pi'])
#plt.legend()
#fig.add_subplot(212)
##plt.hist([width[ele], width[pi]], bins=100, range=(0., 0.05), label=['ele', 'pi'])
#plt.hist([depth[ele], depth[pi]], bins=100, range=(0., 1.), label=['ele', 'pi'])
#plt.legend()
#
#plt.show()
#
#sys.exit(0)

nw = 200
nh = 2

tpr_data = np.empty((nw, nh), dtype=np.float)
fpr_data = np.empty((nw, nh), dtype=np.float)

for iw in range(nw):
    wcut = iw * (0.2 / nw)
    for ih in range(nh):
        hcut = ih * (0.1 / nh)

        tpr_data[iw, ih] = np.sum(np.asarray((truth == 0) & (depth < wcut) & (zrms < hcut), dtype=np.float)) / ele.shape[0]
        fpr_data[iw, ih] = np.sum(np.asarray((truth == 1) & (depth < wcut) & (zrms < hcut), dtype=np.float)) / pi.shape[0]

tpr_data = tpr_data.reshape((nw * nh))
fpr_data = fpr_data.reshape((nw * nh))

#roc_tpr = tpr_data
#roc_fpr = fpr_data

roc_tpr = []
roc_fpr = []
for ient in range(nw * nh):
    #if np.sum(np.asarray((tpr_data < tpr_data[ient]) & (fpr_data > fpr_data[ient]), dtype=np.int)) == 0:
    if np.sum(np.asarray((tpr_data > tpr_data[ient]) & (fpr_data < fpr_data[ient]), dtype=np.int)) == 0:
        roc_tpr.append(tpr_data[ient])
        roc_fpr.append(fpr_data[ient])

roc_tpr = np.array(roc_tpr)
roc_fpr = np.array(roc_fpr)

with h5py.File(output_path, 'w') as out:
    out.create_dataset('tpr', roc_tpr.shape).write_direct(roc_tpr)
    out.create_dataset('fpr', roc_fpr.shape).write_direct(roc_fpr)

plt.scatter(1. - roc_fpr, roc_tpr)
plt.show()
