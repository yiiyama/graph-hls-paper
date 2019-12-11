import os
import sys
import numpy as np
import keras
import uproot
import root_numpy as rnp

import models.simple as garnet

garnet.DEBUG = False

n_vert = 256
n_feat = 4
n_aggregators = 4
n_filters = 4
n_propagate = 4
n_class = 2
n_sample = 10000

x = keras.layers.Input(shape=(n_vert, n_feat))
n = keras.layers.Input(shape=(1,), dtype='int32')
inputs = [x, n]

v = inputs
v = garnet.GarNet(n_aggregators, n_filters, n_propagate, collapse='mean', deduce_nvert=False, discretize_distance=False, name='gar_1')(v)
v = keras.layers.Dense(1, activation='sigmoid')(v)
outputs = v

model = keras.Model(inputs=inputs, outputs=outputs)

model_json = model.to_json()
with open('/afs/cern.ch/user/y/yiiyama/public/hls4ml/garnet-standalone/model.json', 'w') as json_file:
    json_file.write(model_json)

model.load_weights('/afs/cern.ch/user/y/yiiyama/public/hls4ml/garnet-standalone/weights.h5')

tree = uproot.open('/tmp/yiiyama/hgc_gun_nopu_binary_test/1633282_18.root')['tree']
data = tree.arrays(['x', 'n', 'y'], namedecode='ascii')

x = data['x'][:n_sample, :n_vert]
if x.shape[1] < n_vert:
    padding = np.zeros((x.shape[0], n_vert - x.shape[1], x.shape[2]), dtype=np.float32)
    x = np.concatenate((x, padding), axis=1)
n = data['n'][:n_sample]
n = np.minimum(np.ones_like(n) * n_vert, n)
inputs = [x, n]

prob = model.predict(inputs, verbose=1)

if n_class == 2:
    truth = data['y'][:n_sample]
    print('accuracy', np.mean(np.asarray(np.squeeze(np.asarray(prob > 0.5, dtype=np.int32)) == truth, dtype=np.float32)))
else:
    truth = np.argmax(data['y'][:n_sample], axis=1)
    print('accuracy', np.mean(np.asarray(np.argmax(prob, axis=1) == truth, dtype=np.float32)))

print(np.concatenate((prob, np.expand_dims(truth, axis=-1)), axis=-1))

prob = np.squeeze(prob)

if n_class == 2:
    prob_shape = ('prob', np.float32)
else:
    prob_shape = ('prob', np.float32, (n_class,))

entries = np.empty((truth.shape[0],), dtype=[('x', np.float32, (n_vert, n_feat)), ('n', np.int32), prob_shape, ('truth', np.int32)])

for ient, ent in enumerate(zip(x, n, prob, truth)):
    entries[ient] = ent

os.unlink('/tmp/yiiyama/garnet_test.root')
rnp.array2root(entries, '/tmp/yiiyama/garnet_test.root')

in_file = open('/afs/cern.ch/user/y/yiiyama/public/hls4ml/garnet-standalone/garnet-test/tb_data/tb_input_features.dat', 'w')
out_file = open('/afs/cern.ch/user/y/yiiyama/public/hls4ml/garnet-standalone/garnet-test/tb_data/tb_output_predictions.dat', 'w')
truth_file = open('/afs/cern.ch/user/y/yiiyama/public/hls4ml/garnet-standalone/garnet-test/tb_data/tb_input_truth.dat', 'w')

for iline, (xval, nval, pval, tval) in enumerate(zip(x, n, prob, truth)):
    in_file.write(' '.join('%f' % v for v in np.reshape(xval, (-1,))))
    in_file.write(' %d\n' % nval)
    truth_file.write('%d\n' % tval)
    out_file.write('%f\n' % pval)

    #if iline == 200:
    #    break
