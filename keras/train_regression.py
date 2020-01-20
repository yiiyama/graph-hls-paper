#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import keras
import numpy as np

import debug_flag
# Set to True to get printouts
debug_flag.DEBUG = False

from models.regression_simple import make_model, make_loss

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
    parser.add_argument('--train', '-t', metavar='PATH', dest='train_path', nargs='+', help='Training data file.')
    parser.add_argument('--validate', '-v', metavar='PATH', dest='validation_path', nargs='+', help='Validation data file.')
    parser.add_argument('--out', '-o', metavar='PATH', dest='out_path', help='Write HDf5 output to path.')
    parser.add_argument('--ngpu', '-j', metavar='N', dest='ngpu', type=int, default=1, help='Use N GPUs.')
    parser.add_argument('--batch-size', '-b', metavar='N', dest='batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--num-epochs', '-e', metavar='N', dest='num_epochs', type=int, default=80, help='Number of epochs to train over.')
    parser.add_argument('--input-type', '-i', metavar='TYPE', dest='input_type', default='h5', help='Input data format (h5, root, root-sparse).')
    parser.add_argument('--generator', '-g', action='store_true', dest='use_generator', help='Use a generator for input.')
    parser.add_argument('--input-name', '-m', metavar='NAME', dest='input_name', default='events', help='Input dataset (TTree or HDF5 dataset) name.')

    args = parser.parse_args()
    del sys.argv[1:]

    n_vert_max = 1024
    features = None
    y_shape = 1

    model = make_model(n_vert_max, n_feat=4)
    model_single = model
    if args.ngpu > 1:
        model = keras.utils.multi_gpu_model(model_single, args.ngpu)
    
    optimizer = keras.optimizers.Adam(lr=0.00005)
    
    model.compile(optimizer=optimizer, loss=make_loss())

    if args.use_generator:
        if args.input_type == 'h5':
            from generators.h5 import make_generator
        elif args.input_type == 'root':
            from generators.uproot_fixed import make_generator
        elif args.input_type == 'root-sparse':
            from generators.uproot_jagged_keep import make_generator

        train_gen, n_train_steps = make_generator(args.train_path, args.batch_size, features=features, n_vert_max=n_vert_max, y_shape=y_shape, y_dtype=np.float, y_features=[0], dataset_name=args.input_name)
        fit_kwargs = {'steps_per_epoch': n_train_steps, 'epochs': args.num_epochs}

        if args.validation_path:
            valid_gen, n_valid_steps = make_generator(args.validation_path, args.batch_size, features=features, n_vert_max=n_vert_max, y_shape=y_shape, y_dtype=np.float, y_features=[0], dataset_name=args.input_name)
            fit_kwargs['validation_data'] = valid_gen()
            fit_kwargs['validation_steps'] = n_valid_steps

        model.fit_generator(train_gen(), **fit_kwargs)

    else:
        if args.input_type == 'h5':
            import h5py

            ftrain = h5py.File(args.train_path[0])
            inputs = [ftrain['x'], ftrain['n']]
            truth = ftrain['y']
           
            if args.validation_path:
                fvalid = h5py.File(args.validation_path[0])
                val_inputs = [fvalid['x'], fvalid['n']]
                val_truth = fvalid['y']

            shuffle = 'batch'

        elif args.input_type == 'root':
            import uproot

            data = uproot.open(args.train_path[0])['tree'].arrays(['x', 'n', 'y'], namedecode='ascii')
            inputs = [data['x'], data['n']]
            truth = data['y']
            
            if args.validation_path:
                data = uproot.open(args.validation_path[0])['tree'].arrays(['x', 'n', 'y'], namedecode='ascii')
                val_inputs = [data['x'], data['n']]
                val_truth = data['y']

            shuffle = True

        elif args.input_type == 'root-sparse':
            import uproot
            from generators.utils import to_dense
    
            data = uproot.open(args.train_path[0])[args.input_name].arrays(['x', 'n', 'y'], namedecode='ascii')
            inputs = [to_dense(data['n'], data['x'].content, n_vert_max=n_vert_max), data['n'][:]]
            truth = data['y'][:, [0]]

            if args.validation_path:
                data = uproot.open(args.validation_path[0])[args.input_name].arrays(['x', 'n', 'y'], namedecode='ascii')
                val_inputs = [to_dense(data['n'], data['x'].content, n_vert_max=n_vert_max), data['n']]
                val_truth = data['y'][:, [0]]

            shuffle = True

        elif args.input_type == 'random':
            inputs = [np.random.random((args.batch_size, n_vert_max, 4)), np.random.randint(n_vert_max, size=(args.batch_size,))]
            truth = np.random.random((args.batch_size, 1))
            val_inputs = [np.random.random((args.batch_size, n_vert_max, 4)), np.random.randint(n_vert_max, size=(args.batch_size,))]
            val_truth = np.random.random((args.batch_size, 1))

            shuffle = True

        fit_kwargs = {'epochs': args.num_epochs, 'batch_size': args.batch_size, 'shuffle': shuffle}
        if args.validation_path:
            fit_kwargs['validation_data'] = (val_inputs, val_truth)

        model.fit(inputs, truth, **fit_kwargs)
    
    if args.out_path:
        model_single.save(args.out_path)
