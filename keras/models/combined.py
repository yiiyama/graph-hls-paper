import keras
import keras.backend as K
import numpy as np

#from layers.stack import GarNetStack
from layers.caloGraphNN_keras import GarNetStack
from debug_flag import DEBUG

# model parameters
n_vert_max = 256
n_feat = 4
y_features = {'classification': 0, 'regression': 1}
y_dtype = {'classification': np.float, 'regression': np.float}

# training parameters
def sample_weighting(x, y): # for HGCAL dataset; we have more electrons than pions
    w = np.where(y['classification'] < 0.5, 0.613, 0.387)
    return [w, w]

# exported parameters
initial_lr = 0.0005
generator_args = {'n_vert_max': n_vert_max, 'y_features': y_features, 'y_dtype': y_dtype, 'sample_weighting': sample_weighting}
compile_args = {'loss_weights': {'classification': 0.01, 'regression': 0.99}}

def make_model():
    x = keras.layers.Input(shape=(n_vert_max, n_feat))
    n = keras.layers.Input(shape=(1,), dtype='uint16')
    inputs = [x, n]

    #n = keras.layers.Lambda(lambda x: K.print_tensor(x, message='n', summarize=-1))(n)
    v = GarNetStack([4, 4, 8], [8, 8, 16], [8, 8, 16], simplified=True, collapse='mean', input_format='xn', output_activation=None, name='gar_1')([x, n])
    #v = keras.layers.Lambda(lambda x: K.print_tensor(x, message='layer3', summarize=-1))(v)
    v = keras.layers.Dense(16, activation='relu')(v)
    #v = keras.layers.Lambda(lambda x: K.print_tensor(x, message='layer5', summarize=-1))(v)
    v = keras.layers.Dense(8, activation='relu')(v)
    #v = keras.layers.Lambda(lambda x: K.print_tensor(x, message='layer7', summarize=-1))(v)
    v1 = keras.layers.Dense(1, name='regression')(v)
    #v1 = keras.layers.Lambda(lambda x: K.print_tensor(x, message='regression', summarize=-1))(v1)
    v2 = keras.layers.Dense(1, activation='sigmoid', name='classification')(v)
    outputs = [v1, v2]
    
    return keras.Model(inputs=inputs, outputs=outputs)

def make_loss():
    def regression_loss(y_true, y_pred):
        with K.name_scope('regression_loss'):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)

            y_true /= 100. # because our data is max 100 GeV

            if DEBUG:
                y_pred = K.print_tensor(y_pred, message='pred')
                y_true = K.print_tensor(y_true, message='true')

            return K.mean(K.square((y_true - y_pred) / y_true), axis=-1)

    return {'regression': regression_loss, 'classification': 'binary_crossentropy'}
