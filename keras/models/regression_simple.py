from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import keras.backend as K
from layers.caloGraphNN_keras import GarNetStack

from debug_flag import DEBUG

# model parameters
n_vert_max = 1024
n_feat = 4
y_features = 1

# training parameters
input_format = 'xn'

# exported parameters
initial_lr = 0.0005
generator_args = {'n_vert_max': n_vert_max, format=input_format, 'y_features': y_features}

def make_model():
    if input_format == 'xn':
        x = keras.layers.Input(shape=(n_vert_max, n_feat))
    else:
        x = keras.layers.Input(shape=(n_vert_max, n_feat - 1))
        e = keras.layers.Input(shape=(n_vert_max,))
    n = keras.layers.Input(shape=(1,), dtype='uint16')
    if input_format == 'xn':
        inputs = [x, n]
    else:
        inputs = [x, e, n]

    v = inputs
    v = GarNetStack([4, 4], [8, 8], [8, 8], simplified=True, collapse='mean', input_format=input_format, output_activation=None, name='gar_1')(v)
    v = keras.layers.Dense(8, activation='relu')(v)
    v = keras.layers.Dense(1)(v)
    outputs = v
    
    return keras.Model(inputs=inputs, outputs=outputs)

def make_loss():
    def loss(y_true, y_pred):
        with K.name_scope('regression_loss'):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)

            y_true /= 100. # because our data is max 100 GeV

            if DEBUG:
                y_pred = K.print_tensor(y_pred, message='pred')
                y_true = K.print_tensor(y_true, message='true')

            return K.mean(K.square(y_true - y_pred) / y_true, axis=-1)

    return loss
