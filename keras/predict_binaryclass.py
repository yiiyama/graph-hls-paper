#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import numpy as np

from models.binaryclass_simple import make_model
import debug_flag

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

    n_class = 2
    n_vert_max = 256
    #features = list(range(6))
    #features = [0, 1, 2, 3]
    features = None
    n_feat = 4

    # Set to True to get printouts
    debug_flag.DEBUG = False

    model = make_model(n_vert_max, n_feat=n_feat, n_class=n_class)

    model.load_weights(args.weights_path)

    if args.input_type == 'h5':
        from generators.h5 import make_dataset
    elif args.input_type == 'root':
        from generators.uproot_fixed import make_dataset
    elif args.input_type == 'root-sparse':
        from generators.uproot_jagged_keep import make_dataset

    inputs, truth, _ = make_dataset(args.data_path, features=features, n_vert_max=n_vert_max, n_sample=args.n_sample, dataset_name=args.input_name)

    n_sample = inputs[0].shape[0]

    prob = model.predict(inputs, verbose=1)

    if n_class == 2:
        prob = np.squeeze(prob)
        print('accuracy', np.mean(np.asarray(np.asarray(prob > 0.5, dtype=np.int32) == truth, dtype=np.float32)))
    else:
        truth = np.argmax(truth, axis=1)
        print('accuracy', np.mean(np.asarray(np.argmax(prob, axis=1) == truth, dtype=np.float32)))
    
    if args.root_out_path:
        import root_numpy as rnp

        if n_class == 2:
            prob_shape = ('prob', np.float32)
        else:
            prob_shape = ('prob', np.float32, (n_class,))
        
        array = np.empty((n_sample,), dtype=[('x', np.float32, (n_vert_max, n_feat)), ('n', np.int32), prob_shape, ('truth', np.int32)])
        
        for ient, ent in enumerate(zip(*(tuple(inputs) + (prob, truth)))):
            array[ient] = ent

        rnp.array2root(array, args.root_out_path)

    if args.ascii_out_dir:
        in_file = open('%s/tb_input_features.dat' % args.ascii_out_dir, 'w')
        out_file = open('%s/tb_output_predictions.dat' % args.ascii_out_dir, 'w')
        truth_file = open('%s/tb_input_truth.dat' % args.ascii_out_dir, 'w')
        
        for iline, (xval, nval, pval, tval) in enumerate(zip(*(tuple(inputs) + (prob, truth)))):
            in_file.write(' '.join('%f' % v for v in np.reshape(xval, (-1,))))
            in_file.write(' %d\n' % nval)
            truth_file.write('%d\n' % tval)
            out_file.write('%f\n' % pval)
