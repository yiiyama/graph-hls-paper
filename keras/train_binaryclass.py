#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import logging
import glob
import random
import keras
import keras.backend as K
import numpy as np
import uproot
import root_numpy as rnp
import time
import tensorflow as tf

import models.binaryclass_simple as modelmod

if __name__ == '__full__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train the keras model.')
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
    
    tree = uproot.open('/tmp/yiiyama/hgc_gun_nopu_binary_train.root')['tree']
    #tree = uproot.open('/tmp/yiiyama/hgc_gun_nopu_binary_train/1633282_10.root')['tree']
    data = tree.arrays(['x', 'n', 'y'], namedecode='ascii')
    
    inputs = [data['x'], data['n']]
    #inputs = data['x']
    
    truth = data['y']
    #z = np.zeros_like(truth)
    #truth = np.stack((z, truth), axis=-1)
    #truth[:, 0] = np.where(truth[:, 1] == 1, 0, 1)
    
    tree = uproot.open('/tmp/yiiyama/hgc_gun_nopu_binary_train/1633282_9.root')['tree']
    data = tree.arrays(['x', 'n', 'y'], namedecode='ascii')
    
    val_inputs = [data['x'], data['n']]
    val_truth = data['y']
    
    model.fit(inputs, truth, epochs=num_epochs, batch_size=batch_size, validation_data=(val_inputs, val_truth))
    
    #paths = glob.glob('/tmp/yiiyama/hgc_gun_nopu_train/1633282_1*.root')
    #nentries = 2088485
    #generator = make_generator(paths, batch_size=batch_size)
    #model.fit_generator(generator, steps_per_epoch=(nentries / batch_size), epochs=num_epochs, max_queue_size=100)

    if args.out_path:
        model_single.save(args.out_path)
