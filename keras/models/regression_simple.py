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
root_out_dtype = [('x', np.float32, (n_vert_max, n_feat)), ('n', np.int32), ('pred', np.float32), ('truth', np.float32)]
if input_format == 'xen':
    root_out_dtype.insert(1, ('e', np.float32))

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

def evalute_prediction(prediction, truth):
    pred = np.squeeze(prediction)
    truth = np.squeeze(truth) * 1.e-2
    print('sqrt mean ((E_reco - E_gen) / E_gen)^2 =', np.sqrt(np.mean(np.square((pred - truth) / truth))))

def make_root_out_entries(inputs, prediction, truth):
    pred = np.squeeze(prediction)
    truth = np.squeeze(truth) * 1.e-2

    return tuple(inputs) + (pred, truth)

def write_ascii_out(inputs, prediction, in_file, out_file):
    pred = np.squeeze(prediction)

    for entry in zip(*(tuple(inputs) + (pred,))):
        if input_format == 'xn':
            x_val, n_val, p_val = entry
        else:
            x_val, e_val, n_val, p_val = entry
        in_file.write(' '.join('%f' % v for v in np.reshape(x_val, (-1,))))
        if input_format == 'xen':
            in_file.write(' ' + ' '.join('%f' % v for v in e_val))
        in_file.write(' %d\n' % n_val)
        out_file.write('%f\n' % p_val)
