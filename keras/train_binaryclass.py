#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import keras

from models.binaryclass_threelayers import make_model

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

    args = parser.parse_args()
    del sys.argv[1:]

    n_class = 2
    n_vert_max = 256
    #features = list(range(6))
    #features = [0, 1, 2, 3]
    features = None

    model = make_model(n_vert_max, n_feat=4, n_class=n_class)
    model_single = model
    if args.ngpu > 1:
        model = keras.utils.multi_gpu_model(model_single, args.ngpu)
    
    optimizer = keras.optimizers.Adam(lr=0.00005)
    
    if n_class == 2:
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    if args.use_generator:
        if args.input_type == 'h5':
            from generators.h5 import make_generator
        elif args.input_type == 'root':
            from generators.uproot_fixed import make_generator
        elif args.input_type == 'root-sparse':
            import generators.uproot_jagged_keep as generator_mod
            generator_mod.max_cluster_size = n_vert_max
            make_generator = generator_mod.make_generator

        train_gen, n_train_steps = make_generator(args.train_path, args.batch_size, features=features)
        fit_kwargs = {'steps_per_epoch': n_train_steps, 'epochs': args.num_epochs}

        if args.validation_path:
            valid_gen, n_valid_steps = make_generator(args.validation_path, args.batch_size, features=features)
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

        fit_kwargs = {'epochs': args.num_epochs, 'batch_size': args.batch_size, 'shuffle': shuffle}
        if args.validation_path:
            fit_kwargs['validation_data'] = (val_inputs, val_truth)

        model.fit(inputs, truth, **fit_kwargs)
    
    if args.out_path:
        model_single.save(args.out_path)
