#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import numpy as np
import uproot

import models.binaryclass_simple as modelmod

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
    parser.add_argument('--weights', '-w', metavar='PATH', dest='weights_path', help='HDF5 file containing model weights.')
    parser.add_argument('--input', '-i', metavar='PATH', dest='data_path', help='Data file.')
    parser.add_argument('--nsamples', '-n', metavar='N', dest='n_sample', type=int, default=10000, help='Number of samples to process.')
    parser.add_argument('--root-out', '-r', metavar='PATH', dest='root_out_path', help='Write prediction results to a ROOT file.')
    parser.add_argument('--ascii-out', '-a', metavar='PATH', dest='ascii_out_dir', help='Write prediction results to ASCII files in a directory.')

    args = parser.parse_args()
    del sys.argv[1:]

    # Set to True to get printouts
    modelmod.DEBUG = False

    model = modelmod.model

    model.load_weights(args.weights_path)

    tree = uproot.open(args.data_path)['tree']
    data = tree.arrays(['x', 'n', 'y'], namedecode='ascii')

    n_sample = args.n_sample
    if n_sample < data['x'].shape[0]:
        x = data['x'][:n_sample]
        n = data['n'][:n_sample]
        inputs = [x, n]
    else:
        n_sample = data['x'].shape[0]

    prob = model.predict(inputs, verbose=1)

    if modelmod.n_class == 2:
        truth = data['y'][:n_sample]
        prob = np.squeeze(prob)
        print('accuracy', np.mean(np.asarray(np.asarray(prob > 0.5, dtype=np.int32) == truth, dtype=np.float32)))
    else:
        truth = np.argmax(data['y'][:n_sample], axis=1)
        print('accuracy', np.mean(np.asarray(np.argmax(prob, axis=1) == truth, dtype=np.float32)))
    
    if args.root_out_path:
        import root_numpy as rnp

        if modelmod.n_class == 2:
            prob_shape = ('prob', np.float32)
        else:
            prob_shape = ('prob', np.float32, (modelmod.n_class,))
        
        entries = np.empty((n_sample,), dtype=[('x', np.float32, x.shape[1:]), ('n', np.int32), prob_shape, ('truth', np.int32)])
        
        for ient, ent in enumerate(zip(x, n, prob, truth)):
            entries[ient] = ent

        rnp.array2root(entries, args.root_out_path)

    if args.ascii_out_dir:
        in_file = open('%s/tb_input_features.dat' % args.ascii_out_dir, 'w')
        out_file = open('%s/tb_output_predictions.dat' % args.ascii_out_dir, 'w')
        truth_file = open('%s/tb_input_truth.dat' % args.ascii_out_dir, 'w')
        
        for iline, (xval, nval, pval, tval) in enumerate(zip(x, n, prob, truth)):
            in_file.write(' '.join('%f' % v for v in np.reshape(xval, (-1,))))
            in_file.write(' %d\n' % nval)
            truth_file.write('%d\n' % tval)
            out_file.write('%f\n' % pval)

