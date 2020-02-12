#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import math
import keras
import numpy as np

argv = list(sys.argv)
del sys.argv[1:]

import debug_flag
# Set to True to get printouts
debug_flag.DEBUG = False

from models.regression_simple import make_model, make_loss, input_format
from callbacks.plot_distribution import PlotDistribution

if __name__ == '__main__':
    sys.argv = argv

    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
    parser.add_argument('--train', '-t', metavar='PATH', dest='train_path', nargs='+', help='Training data file.')
    parser.add_argument('--validate', '-v', metavar='PATH', dest='validation_path', nargs='+', help='Validation data file.')
    parser.add_argument('--predict', '-p', metavar='PATH', dest='pred_path', help='Run on-the-fly prediction on test data.')
    parser.add_argument('--out', '-o', metavar='PATH', dest='out_path', help='Write HDf5 output to path.')
    parser.add_argument('--ngpu', '-j', metavar='N', dest='ngpu', type=int, default=1, help='Use N GPUs.')
    parser.add_argument('--batch-size', '-b', metavar='N', dest='batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--num-epochs', '-e', metavar='N', dest='num_epochs', type=int, default=80, help='Number of epochs to train over.')
    parser.add_argument('--num-steps', '-s', metavar='N', dest='num_steps', type=int, default=None, help='Number of steps per epoch.')
    parser.add_argument('--input-type', '-i', metavar='TYPE', dest='input_type', default='h5', help='Input data format (h5, root, root-sparse).')
    parser.add_argument('--generator', '-g', action='store_true', dest='use_generator', help='Use a generator for input.')
    parser.add_argument('--input-name', '-m', metavar='NAME', dest='input_name', default='events', help='Input dataset (TTree or HDF5 dataset) name.')

    args = parser.parse_args()
    del sys.argv[1:]

    n_vert_max = 1024
    features = None
    y_features = [3]

    n_feat = 4 if features is None else len(features)

    model = make_model(n_vert_max, n_feat=n_feat)
    model_single = model
    if args.ngpu > 1:
        model = keras.utils.multi_gpu_model(model_single, args.ngpu)
    
    optimizer = keras.optimizers.Adam(lr=0.0005)
    
    model.compile(optimizer=optimizer, loss=make_loss())

    callbacks = [
        #keras.callbacks.LearningRateScheduler(lambda epoch: 5.e-4 * math.exp(-0.03 * epoch) * math.cos(math.pi / 5. * epoch) + 5.e-4, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, verbose=1),
        keras.callbacks.EarlyStopping(verbose=1, patience=3)
    ]
    if args.out_path:
        callbacks.append(keras.callbacks.ModelCheckpoint(args.out_path))
    if args.pred_path:
        callbacks.append(PlotDistribution(args.pred_path, args.input_type, n_vert_max, features=features, input_name=args.input_name))

    if args.use_generator:
        if args.input_type == 'h5':
            from generators.h5 import make_generator
        elif args.input_type == 'root':
            from generators.uproot_fixed import make_generator
        elif args.input_type == 'root-sparse':
            from generators.uproot_jagged_keep import make_generator

        train_gen, n_train_steps = make_generator(args.train_path, args.batch_size, format=input_format, features=features, n_vert_max=n_vert_max, y_dtype=np.float, y_features=y_features, dataset_name=args.input_name)

        fit_kwargs = {'epochs': args.num_epochs, 'callbacks': callbacks, 'shuffle': True}
        if n_train_steps is not None:
            fit_kwargs['steps_per_epoch'] = n_train_steps

        if args.validation_path:
            valid_gen, n_valid_steps = make_generator(args.validation_path, args.batch_size, format=input_format, features=features, n_vert_max=n_vert_max, y_dtype=np.float, y_features=y_features, dataset_name=args.input_name)

            fit_kwargs['validation_data'] = valid_gen
            if n_valid_steps is not None:
                fit_kwargs['validation_steps'] = n_valid_steps

        if args.num_steps is not None:
            fit_kwargs['steps_per_epoch'] = args.num_steps

        model.fit_generator(train_gen, **fit_kwargs)

    else:
        if args.input_type == 'random':
            inputs = [np.random.random((args.batch_size, n_vert_max, 4)), np.random.randint(n_vert_max, size=(args.batch_size,))]
            truth = np.random.random((args.batch_size, 1))
            val_inputs = [np.random.random((args.batch_size, n_vert_max, 4)), np.random.randint(n_vert_max, size=(args.batch_size,))]
            val_truth = np.random.random((args.batch_size, 1))

            shuffle = True

        else:
            if args.input_type == 'h5':
                from generators.h5 import make_dataset
            elif args.input_type == 'root':
                from generators.uproot_fixed import make_dataset
            elif args.input_type == 'root-sparse':
                from generators.uproot_jagged_keep import make_dataset
    
            inputs, truth, shuffle = make_dataset(args.train_path[0], format=input_format, features=features, n_vert_max=n_vert_max, y_features=y_features, dataset_name=args.input_name)

            if args.validation_path:
                val_inputs, val_truth, _ = make_dataset(args.validation_path[0], format=input_format, features=features, n_vert_max=n_vert_max, y_features=y_features, dataset_name=args.input_name)

        fit_kwargs = {'epochs': args.num_epochs, 'batch_size': args.batch_size, 'shuffle': shuffle, 'callbacks': callbacks}
        if args.validation_path:
            fit_kwargs['validation_data'] = (val_inputs, val_truth)

        if args.num_steps is not None:
            n = args.num_steps * args.batch_size
            inputs = [inputs[0][:n], inputs[1][:n]]
            truth = truth[:n]

        model.fit(inputs, truth, **fit_kwargs)
    
    if args.out_path:
        model_single.save(args.out_path)
