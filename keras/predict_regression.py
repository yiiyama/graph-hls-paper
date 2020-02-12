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
    y_features = [3]

    n_feat = 4 if features is None else len(features)

    model = make_model(n_vert_max, n_feat=n_feat)

    model.load_weights(args.weights_path)

    if args.input_type == 'h5':
        from generators.h5 import make_dataset
    elif args.input_type == 'root':
        from generators.uproot_fixed import make_dataset
    elif args.input_type == 'root-sparse':
        from generators.uproot_jagged_keep import make_dataset

    inputs, truth, _ = make_dataset(args.data_path, format=input_format, features=features, n_vert_max=n_vert_max, y_features=y_features, n_sample=args.n_sample, dataset_name=args.input_name)

    n_sample = inputs[0].shape[0]

    pred = np.squeeze(model.predict(inputs, verbose=1))

    truth = np.squeeze(truth) * 1.e-2
    print('mean (E_reco - E_gen)^2 / E_gen =', np.mean(np.square((pred - truth) / truth)))

    if args.root_out_path:
        import root_numpy as rnp

        dtype = [('x', np.float32, (n_vert_max, n_feat)), ('n', np.int32), ('pred', np.float32), ('truth', np.float32)]
        if input_format == 'xen':
            dtype.insert(1, ('e', np.float32))
        array = np.empty((n_sample,), dtype=dtype)
        
        for ient, ent in enumerate(zip(*(tuple(inputs) + (pred, truth)))):
            array[ient] = ent

        rnp.array2root(array, args.root_out_path, mode='recreate')

    if args.ascii_out_dir:
        in_file = open('%s/tb_input_features.dat' % args.ascii_out_dir, 'w')
        out_file = open('%s/tb_output_predictions.dat' % args.ascii_out_dir, 'w')
        truth_file = open('%s/tb_input_truth.dat' % args.ascii_out_dir, 'w')

        for iline, entry in enumerate(zip(*(tuple(inputs) + (pred, truth)))):
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
