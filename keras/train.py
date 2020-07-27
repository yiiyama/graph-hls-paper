#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import math
import importlib
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description='Train the keras model.')
parser.add_argument('model', metavar='MODEL', help='Name of the module in the models directory.')
parser.add_argument('--train', '-t', metavar='PATH', dest='train_path', nargs='+', help='Training data file.')
parser.add_argument('--validate', '-v', metavar='PATH', dest='validation_path', nargs='+', help='Validation data file.')
parser.add_argument('--weights', '-w', metavar='PATH', dest='weights_path', help='Read weights from HDF5.')
parser.add_argument('--out', '-o', metavar='PATH', dest='out_path', help='Write HDF5 output to path.')
parser.add_argument('--ngpu', '-j', metavar='N', dest='ngpu', type=int, default=1, help='Use N GPUs.')
parser.add_argument('--gpus', '-g', metavar='ID', dest='gpus', type=int, nargs='+', help='Use specified GPUs (overrides --ngpu).')
parser.add_argument('--batch-size', '-b', metavar='N', dest='batch_size', type=int, default=512, help='Batch size.')
parser.add_argument('--num-epochs', '-e', metavar='N', dest='num_epochs', type=int, default=80, help='Number of epochs to train over.')
parser.add_argument('--num-steps', '-s', metavar='N', dest='num_steps', type=int, default=None, help='Number of steps per epoch.')
parser.add_argument('--input-type', '-i', metavar='TYPE', dest='input_type', default='h5', help='Input data format (h5, root, root-sparse).')
parser.add_argument('--input-name', '-m', metavar='NAME', dest='input_name', default='events', help='Input dataset (TTree or HDF5 dataset) name.')

args = parser.parse_args()
del sys.argv[1:]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if args.gpus is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join('%d' % u for u in args.gpus)
    args.ngpu = len(args.gpus)

try:
    import tensorflow.keras as keras
except ImportError:
    import keras

if hasattr(keras.backend, 'tensorflow_backend'):
    import tensorflow as tf
    
    if float(tf.__version__[:tf.__version__.rfind('.')]) > 2.:
        tfback = keras.backend.tensorflow_backend
    
        def _get_available_gpus():
            """Get a list of available gpu devices (formatted as strings).
        
            # Returns
                A list of available GPU devices.
            """
            #global _LOCAL_DEVICES
            if tfback._LOCAL_DEVICES is None:
                devices = tf.config.list_logical_devices()
                tfback._LOCAL_DEVICES = [x.name for x in devices]
            return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
        
        tfback._get_available_gpus = _get_available_gpus

import debug_flag
# Set to True to get printouts
debug_flag.DEBUG = False

modelmod = importlib.import_module('models.%s' % args.model)

model = modelmod.make_model()

if args.weights_path is not None:
    model.load_weights(args.weights_path)

model_single = model
if args.ngpu > 1:
    model = keras.utils.multi_gpu_model(model_single, args.ngpu)

try:
    initial_lr = modelmod.initial_lr
except AttributeError:
    initial_lr = 0.0005

optimizer = keras.optimizers.Adam(lr=initial_lr)

try:
    compile_args = modelmod.compile_args
except AttributeError:
    compile_args = dict()

model.compile(optimizer=optimizer, loss=modelmod.make_loss(), **compile_args)

callbacks = [
    #keras.callbacks.LearningRateScheduler(lambda epoch: 5.e-4 * math.exp(-0.03 * epoch) * math.cos(math.pi / 5. * epoch) + 5.e-4, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, verbose=1),
    keras.callbacks.EarlyStopping(verbose=1, patience=10)
]
if args.out_path:
    callbacks.append(keras.callbacks.ModelCheckpoint(args.out_path))

if args.input_type == 'h5':
    from generators.h5 import make_generator
elif args.input_type == 'root':
    from generators.uproot_fixed import make_generator
elif args.input_type == 'root-sparse':
    from generators.uproot_jagged_keep import make_generator

train_gen, n_train_steps = make_generator(args.train_path, args.batch_size, num_steps=args.num_steps, dataset_name=args.input_name, **modelmod.generator_args)

fit_kwargs = {'epochs': args.num_epochs, 'callbacks': callbacks, 'shuffle': True}
if n_train_steps is not None:
    fit_kwargs['steps_per_epoch'] = n_train_steps

if args.validation_path:
    #from generators.utils import make_dataset

    valid_gen, n_valid_steps = make_generator(args.validation_path, 1024, dataset_name=args.input_name, **modelmod.generator_args)

    # feeding a generator as validation_data appears to cause wrong validation loss reporting (keras computing mean of batch validation losses?)
    #fit_kwargs['validation_data'] = make_dataset(valid_gen, n_valid_steps)
    fit_kwargs['validation_data'] = valid_gen

if args.num_steps is not None:
    fit_kwargs['steps_per_epoch'] = args.num_steps

try:
    model.fit(train_gen, **fit_kwargs)
    #model.fit(x=x, y=y, **fit_kwargs)
except KeyboardInterrupt:
    pass

if args.out_path:
    model_single.save(args.out_path)
