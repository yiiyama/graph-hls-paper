import numpy as np
try:
    import tensorflow.keras as keras
except ImportError:
    import keras
K = keras.backend

from layers.weighted import ZWeightedSum
from debug_flag import DEBUG

# model parameters
n_vert_max = 128
n_feat = 2
y_features = 1
y_dtype = np.float

# training parameters
#def sample_weighting(x, y): # for HGCAL dataset; we have more electrons than pions
#    w = np.where(y['classification'] < 0.5, 0.613, 0.387)
#    return [w, w]

# exported parameters
initial_lr = 0.0005
generator_args = {'n_vert_max': n_vert_max, 'format': 'x', 'features': [2, 3], 'y_features': y_features, 'y_dtype': y_dtype}
root_out_dtype = [('x', np.float32, (n_vert_max, n_feat)), ('n', np.int32), ('prediction', np.float32), ('truth', np.float32)]

def make_model():
    x = keras.layers.Input(shape=(n_vert_max, n_feat))
    inputs = x

    v = ZWeightedSum()(inputs)

    outputs = v
    
    return keras.Model(inputs=inputs, outputs=outputs)

def make_loss():
    def regression_loss(y_true, y_pred):
        with K.name_scope('regression_loss'):
            #if not K.is_tensor(y_pred):
            #    y_pred = K.constant(y_pred)

            y_true /= 100. # because our data is max 100 GeV

            if DEBUG:
                y_pred = K.print_tensor(y_pred, message='pred')
                y_true = K.print_tensor(y_true, message='true')

            return K.mean(K.square((y_true - y_pred) / y_true), axis=-1)

    return regression_loss

def evaluate_prediction(prediction, truth):
    prediction = np.squeeze(prediction)

    truth = np.squeeze(truth) * 1.e-2
    print('sqrt (mean ((E_reco - E_gen) / E_gen)^2) =', np.sqrt(np.mean(np.square((prediction - truth) / truth))))

def make_root_out_entries(inputs, prediction, truth):
    prediction = np.squeeze(prediction)
    truth = np.squeeze(truth) * 1.e-2

    return tuple(inputs) + (prediction, truth)

def make_h5_out_data(prediction, truth):
    prediction = np.squeeze(prediction)
    truth = np.squeeze(truth) * 1.e-2

    return {'prediction': prediction, 'truth': truth}

def write_ascii_out(inputs, prediction, truth, in_file, out_file, truth_file):
    pass
