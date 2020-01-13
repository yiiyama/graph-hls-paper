#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import keras
import uproot
import h5py

import models.binaryclass_simple as modelmod

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
    parser.add_argument('--train', '-t', metavar='PATH', dest='train_path', nargs='+', help='Training data file.')
    parser.add_argument('--validate', '-v', metavar='PATH', dest='validation_path', nargs='+', help='Validation data file.')
    parser.add_argument('--out', '-o', metavar='PATH', dest='out_path', help='Write HDf5 output to path.')
    parser.add_argument('--ncpu', '-j', metavar='N', dest='ncpu', type=int, default=1, help='Write HDf5 output to path.')
    parser.add_argument('--batch-size', '-b', metavar='N', dest='batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--num-epochs', '-e', metavar='N', dest='num_epochs', type=int, default=80, help='Number of epochs to train over.')
    parser.add_argument('--sparse-input', '-S', action='store_true', dest='sparse_input', help='Input is variable length.')
    parser.add_argument('--input-type', '-i', metavar='TYPE', dest='input_type', help='Input data format (h5, root, root-sparse).')
    parser.add_argument('--generator', '-g', action='store_true', dest='use_generator', help='Use a generator for input.')

    args = parser.parse_args()
    del sys.argv[1:]

    model = modelmod.model
    model_single = model
    if args.ncpu > 1:
        model = keras.utils.multi_gpu_model(model_single, args.ncpu)
    
    optimizer = keras.optimizers.Adam(lr=0.00005)
    
    if modelmod.n_class == 2:
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    if args.use_generator:
        if args.input_type == 'h5':
            from generators.h5 import make_generator
        elif args.input_type == 'root':
            from generators.uproot import make_generator
        elif args.input_type == 'root-sparse':
            from generators.uproot_jagged import make_generator

        train_gen, n_train_steps = make_generator(args.train_path, args.batch_size)
        valid_gen, n_valid_steps = make_generator(args.validation_path, args.batch_size)

        model.fit_generator(
            train_gen(),
            steps_per_epoch=n_train_steps,
            epochs=args.num_epochs,
            validation_data=valid_gen(),
            validation_steps=n_valid_steps
        )

    else:
        if args.input_type == 'h5':
            ftrain = h5py.File(args.train_path)
            inputs = [ftrain['x'], ftrain['n']]
            truth = ftrain['y']
        
            fvalid = h5py.File(args.validation_path)
            val_inputs = [fvalid['x'], fvalid['n']]
            val_truth = fvalid['y']

            shuffle = 'batch'

        elif args.input_type == 'root':
            data = uproot.open(args.train_path)['tree'].arrays(['x', 'n', 'y'], namedecode='ascii')
            inputs = [data['x'], data['n']]
            truth = data['y']
            
            data = uproot.open(args.validation_path)['tree'].arrays(['x', 'n', 'y'], namedecode='ascii')
            val_inputs = [data['x'], data['n']]
            val_truth = data['y']

            shuffle = True
        
        model.fit(
            inputs,
            truth,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            validation_data=(val_inputs, val_truth),
            shuffle=shuffle
        )
    
    if args.out_path:
        model_single.save(args.out_path)
