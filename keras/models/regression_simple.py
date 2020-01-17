from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import keras.backend as K
from layers.simple import GarNet

def make_model(n_vert, n_feat):
    n_aggregators = 4
    n_filters = 4
    n_propagate = 4
    
    x = keras.layers.Input(shape=(n_vert, n_feat))
    n = keras.layers.Input(shape=(1,), dtype='int32')
    inputs = [x, n]
    
    v = inputs
    v = GarNet(n_aggregators, n_filters, n_propagate, collapse='mean', deduce_nvert=False, name='gar_1')(v)
    v = keras.layers.Dense(4, activation='tanh')(v)
    v = keras.layers.Dense(1)(v)
    outputs = v
    
    return keras.Model(inputs=inputs, outputs=outputs)

def make_loss():
    def loss(y_true, y_pred):
        with K.name_scope('regression_loss'):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
    
            y_true = K.cast(y_true, y_pred.dtype)
            return K.square(y_true - y_pred) / y_true

    return loss

if __name__ == '__main__':
    import sys

    out_path = sys.argv[1]
    n_vert = int(sys.argv[2])
    n_feat = int(sys.argv[3])

    model = make_model(n_vert, n_feat)

    with open(out_path, 'w') as json_file:
        json_file.write(model.to_json())
