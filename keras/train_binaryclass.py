#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import keras
import numpy as np
import uproot

import models.binaryclass_simple as modelmod

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
    parser.add_argument('--train', '-t', metavar='PATH', dest='train_path', help='Training data file.')
    parser.add_argument('--validate', '-v', metavar='PATH', dest='validation_path', help='Validation data file.')
    parser.add_argument('--out', '-o', metavar='PATH', dest='out_path', help='Write HDf5 output to path.')
    parser.add_argument('--ncpu', '-j', metavar='N', dest='ncpu', type=int, default=1, help='Write HDf5 output to path.')

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
    
    batch_size = 512
    num_epochs = 80
    
    tree = uproot.open(args.train_path)['tree']
    data = tree.arrays(['x', 'n', 'y'], namedecode='ascii')
    
    inputs = [data['x'], data['n']]
    truth = data['y']
    
    tree = uproot.open(args.validation_path)['tree']
    data = tree.arrays(['x', 'n', 'y'], namedecode='ascii')
    
    val_inputs = [data['x'], data['n']]
    val_truth = data['y']
    
    model.fit(inputs, truth, epochs=num_epochs, batch_size=batch_size, validation_data=(val_inputs, val_truth))
    
    if args.out_path:
        model_single.save(args.out_path)
