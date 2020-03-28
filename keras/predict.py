#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import importlib
import numpy as np

import debug_flag
# Set to True to get printouts
debug_flag.DEBUG = False

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
    parser.add_argument('weights_path', metavar='PATH', help='HDF5 file containing model weights.')
    parser.add_argument('data_path', metavar='PATH', help='Data file.')
    parser.add_argument('--input-type', '-i', metavar='TYPE', dest='input_type', default='h5', help='Input data format (h5, root, root-sparse).')
    parser.add_argument('--batch-size', '-b', metavar='N', dest='batch_size', type=int, default=32, help='Input batch size.')
    parser.add_argument('--num-batches', '-s', metavar='N', dest='num_batches', type=int, default=None, help='Number of batches to process.')
    parser.add_argument('--root-out', '-r', metavar='PATH', dest='root_out_path', help='Write prediction results to a ROOT file.')
    parser.add_argument('--ascii-out', '-a', metavar='PATH', dest='ascii_out_dir', help='Write prediction results to ASCII files in a directory.')
    parser.add_argument('--input-name', '-m', metavar='NAME', dest='input_name', default='events', help='Input dataset (TTree or HDF5 dataset) name.')

    args = parser.parse_args()
    del sys.argv[1:]

    modelmod = importlib.import_module('models.%s' % args.model)

    model = make_model()
    model.load_weights(args.weights_path)

    if args.input_type == 'h5':
        from generators.h5 import make_generator
    elif args.input_type == 'root':
        from generators.uproot_fixed import make_generator
    elif args.input_type == 'root-sparse':
        from generators.uproot_jagged_keep import make_generator

    inputs, truth, _ = make_generator([args.data_path], args.batch_size, num_steps=args.num_batches, dataset_name=args.input_name, **modelmod.generator_args)
    
    prediction = model.predict(inputs, verbose=1)

    modelmod.evaluate_prediction(prediction, truth)

    if args.root_out_path:
        import root_numpy as rnp
        import uproot

        try:
            os.makedirs(os.path.dirname(args.root_out_path))
        except OSError:
            pass

        dtype = modelmod.root_out_dtype
        entries = modelmod.make_root_out_entries(inputs, prediction, truth)
        
        ## more info for the HGCAL dataset
        #dtype += [('cl_pt', np.float32), ('gen_eta', np.float32)]
        #data = uproot.open(args.data_path)[args.input_name].arrays(['cl_pt', 'gen_eta'], namedecode='ascii')
        #entries += (data['cl_pt'], data['gen_eta'])

        array = np.empty((n_sample,), dtype=dtype)
        
        for ient, ent in enumerate(zip(*entries)):
            array[ient] = ent

        rnp.array2root(array, args.root_out_path, mode='recreate')

    if args.ascii_out_dir:
        try:
            os.makedirs(args.ascii_out_dir)
        except OSError:
            pass
            
        in_file = open('%s/tb_input_features.dat' % args.ascii_out_dir, 'w')
        out_file = open('%s/tb_output_predictions.dat' % args.ascii_out_dir, 'w')

        modelmod.write_ascii_out(inputs, prediction, in_file, out_file)
