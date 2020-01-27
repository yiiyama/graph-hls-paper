#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import numpy as np

import debug_flag
# Set to True to get printouts
debug_flag.DEBUG = False

from models.regression_simple import make_model, input_format

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
    parser.add_argument('weights_path', metavar='PATH', help='HDF5 file containing model weights.')
    parser.add_argument('data_path', metavar='PATH', help='Data file.')
    parser.add_argument('--input-type', '-i', metavar='TYPE', dest='input_type', default='h5', help='Input data format (h5, root, root-sparse).')
    parser.add_argument('--nsamples', '-n', metavar='N', dest='n_sample', type=int, default=10000, help='Number of samples to process.')
    parser.add_argument('--root-out', '-r', metavar='PATH', dest='root_out_path', help='Write prediction results to a ROOT file.')
    parser.add_argument('--ascii-out', '-a', metavar='PATH', dest='ascii_out_dir', help='Write prediction results to ASCII files in a directory.')
    parser.add_argument('--input-name', '-m', metavar='NAME', dest='input_name', default='events', help='Input dataset (TTree or HDF5 dataset) name.')

    args = parser.parse_args()
    del sys.argv[1:]

    n_vert_max = 1024
    features = None
    y_features = [0]

    n_feat = 4 if features is None else len(features)
    y_shape = None if y_features is None else len(y_features)

    model = make_model(n_vert_max, n_feat=n_feat)

    model.load_weights(args.weights_path)

    if args.input_type == 'h5':
        import h5py

        data = h5py.File(args.data_path)

    elif args.input_type == 'root':
        import uproot

        data = uproot.open(args.data_path)[args.input_name].arrays(['x', 'n', 'y'], namedecode='ascii')
        
    elif args.input_type == 'root-sparse':
        import uproot
        from generators.utils import to_dense

        data = uproot.open(args.data_path)[args.input_name].arrays(['x', 'n', 'y'], namedecode='ascii')
        data['x'] = to_dense(data['n'], data['x'].content, n_vert_max=n_vert_max)

    if input_format == 'xn':
        x = data['x']
    else:
        x = data['x'][:, :, :3]
        e = data['x'][:, :, 3]
    n = data['n']
    y = data['y']

    if features is not None:
        x = x[:, :, features]

    if y_features is not None:
        y = y[:, y_features]

    n_sample = args.n_sample
    if n_sample < n.shape[0]:
        x = x[:n_sample]
        if input_format == 'xen':
            e = e[:n_sample]
        n = n[:n_sample]
    else:
        n_sample = n.shape[0]

    if input_format == 'xn':
        inputs = [x, n]
    else:
        inputs = [x, e, n]

    pred = np.squeeze(model.predict(inputs, verbose=1))

    truth = np.squeeze(y[:n_sample]) * 1.e-2
    print('mean (E_reco - E_gen)^2 / E_gen =', np.mean(np.square((pred - truth) / truth)))

    if input_format == 'xn':
        entries = zip(x, n, pred, truth)
    else:
        entries = zip(x, e, n, pred, truth)

    if args.root_out_path:
        import root_numpy as rnp

        dtype = [('x', np.float32, x.shape[1:]), ('n', np.int32), ('pred', np.float32), ('truth', np.float32)]
        if input_format == 'xen':
            dtype.insert(1, ('e', np.float32))
        entries = np.empty((n_sample,), dtype=dtype)
        
        for ient, ent in enumerate(entries):
            entries[ient] = ent

        rnp.array2root(entries, args.root_out_path)

    if args.ascii_out_dir:
        in_file = open('%s/tb_input_features.dat' % args.ascii_out_dir, 'w')
        out_file = open('%s/tb_output_predictions.dat' % args.ascii_out_dir, 'w')
        truth_file = open('%s/tb_input_truth.dat' % args.ascii_out_dir, 'w')

        for iline, entry in enumerate(entries):
            if input_format == 'xn':
                x_val, n_val, p_val, t_val = entry
            else:
                x_val, e_val, n_val, p_val, t_val = entry
            in_file.write(' '.join('%f' % v for v in np.reshape(x_val, (-1,))))
            if input_format == 'xen':
                in_file.write(' ' + ' '.join('%f' % v for v in e_val))
            in_file.write(' %d\n' % n_val)
            truth_file.write('%f\n' % t_val)
            out_file.write('%f\n' % p_val)
