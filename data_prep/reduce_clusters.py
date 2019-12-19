#!/usr/bin/env python

######################################################################################
## Quick converter to reduce the HGCalTP/extractor output to bare minimum necessary
## for egamma / hadron binary classification
## Usage:
##  reduce_clusters.py input_file output_directory
######################################################################################

import os
import sys
import shutil
import numpy as np
import uproot
import root_numpy as rnp

n_vert = 256

branches = [
    'cluster_pt',
    'n_cell'
]

x_branches = [
    'cell_energy',
    'cell_theta',
    'cell_phi',
    'cell_z'
]

y_branches = [
    'electron',
    'muon',
    'photon',
    'pi0',
    'neutral',
    'charged'
]

path, out_dir = sys.argv[1:3]

#path = 'root://eoscms.cern.ch/' + path

tree = uproot.open(path)['clusters']
data = tree.arrays(branches + x_branches + y_branches, namedecode='ascii')

# Cut out clusters with pt < 5 GeV or truth == muon
event_filter = (data['cluster_pt'] > 5.)
event_filter = (data['muon'][event_filter] == 0)

# Three branches: n, x, and y

n = data['n_cell'][event_filter]

x = np.stack(tuple(data[b][event_filter] for b in x_branches), axis=-1)

x[:, :, 0] = np.sqrt(x[:, :, 0]) # take the sqrt of energy
x[:, :, 3] = np.where(x[:, :, 3] == 0., 0., (np.abs(x[:, :, 3]) - 300.) / 200.) # shift and scale z to O(1)

if x.shape[1] < n_vert:
    padding = np.zeros((x.shape[0], n_vert - x.shape[1], x.shape[2]), dtype=np.float32)
    x = np.concatenate((x, padding), axis=1)

egamma = data['electron'] + data['photon'] + data['pi0']

y = egamma[event_filter]

# Write to tree

entries = np.empty((n.shape[0],), dtype=[('x', np.float32, (n_vert, 4)), ('n', np.int16), ('y', np.int8)])

for ient, ent in enumerate(zip(x, n, y)):
    entries[ient] = ent

fname = os.path.basename(path)
tmp_out = '%s/%s' % (os.getenv('TMPDIR', '/tmp'), fname)

rnp.array2root(entries, tmp_out)

out_path = '%s/%s' % (out_dir, fname)
if tmp_out != out_path:
    shutil.move(tmp_out, out_path)
