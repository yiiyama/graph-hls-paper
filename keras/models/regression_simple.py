from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import keras.backend as K
from layers.simple import GarNet

from debug_flag import DEBUG

input_format = 'xn'

def make_model(n_vert, n_feat):
    n_aggregators = 4
    n_filters = 4
    n_propagate = 4
    
    if input_format == 'xn':
        x = keras.layers.Input(shape=(n_vert, n_feat))
    else:
        x = keras.layers.Input(shape=(n_vert, n_feat - 1))
        e = keras.layers.Input(shape=(n_vert,))
    n = keras.layers.Input(shape=(1,), dtype='uint16')
    if input_format == 'xn':
        inputs = [x, n]
    else:
        inputs = [x, e, n]

    v = inputs
    v = GarNet(4, 8, 8, input_format=input_format, name='gar_1')(v)
    v = GarNet(4, 8, 8, input_format=input_format, name='gar_2')([v, n])
    v = GarNet(4, 8, 8, input_format=input_format, name='gar_3')([v, n])
#    v = GarNet(4, 8, 8, input_format=input_format, name='gar_4')([v, n])
#    v = GarNet(4, 8, 8, input_format=input_format, name='gar_5')([v, n])
    v = GarNet(4, 8, 8, collapse='mean', input_format=input_format, name='gar_6')([v, n])
    v = keras.layers.Dense(8, activation='relu')(v)
    #v = keras.layers.Dense(4, activation='relu')(v)
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

if __name__ == '__main__':
    import sys

    out_path = sys.argv[1]
    n_vert = int(sys.argv[2])
    n_feat = int(sys.argv[3])

    model = make_model(n_vert, n_feat)

    with open(out_path, 'w') as json_file:
        json_file.write(model.to_json())
