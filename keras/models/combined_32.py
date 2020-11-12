import numpy as np
try:
    import tensorflow.keras as keras
except ImportError:
    import keras
K = keras.backend

from layers.caloGraphNN.caloGraphNN_keras import GarNetStack
from debug_flag import DEBUG

# model parameters
n_vert_max = 32
n_feat = 4
y_features = {'classification': 0, 'regression': 1}
y_dtype = {'classification': np.float, 'regression': np.float}

# training parameters
#def sample_weighting(x, y): # for HGCAL dataset; we have more electrons than pions
#    w = np.where(y['classification'] < 0.5, 0.613, 0.387)
#    return [w, w]

# exported parameters
initial_lr = 0.0005
generator_args = {'n_vert_restrict': n_vert_max, 'y_features': y_features, 'y_dtype': y_dtype}
#generator_args['sample_weighting'] = sample_weighting
compile_args = {'loss_weights': {'classification': 0.01, 'regression': 0.99}}
root_out_dtype = [('x', np.float32, (n_vert_max, n_feat)), ('n', np.int32), ('pred_regression', np.float32), ('pred_classification', np.float32), ('truth_regression', np.float32), ('truth_classification', np.float32)]

def _make_model(quantize):
    x = keras.layers.Input(shape=(n_vert_max, n_feat))
    n = keras.layers.Input(shape=(1,), dtype='uint16')
    inputs = [x, n]

    #n = keras.layers.Lambda(lambda x: K.print_tensor(x, message='n', summarize=-1))(n)
    v = GarNetStack([4, 4, 8], [8, 8, 16], [8, 8, 16], simplified=True, collapse='mean', input_format='xn', output_activation=None, name='gar_1', quantize_transforms=quantize)([x, n])
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

def make_model():
    return _make_model(False)

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

    return {'regression': regression_loss, 'classification': 'binary_crossentropy'}

def evaluate_prediction(prediction, truth):
    pred_regression, pred_classification = map(np.squeeze, prediction)

    truth_regression = np.squeeze(truth['regression']) * 1.e-2
    print('sqrt (mean ((E_reco - E_gen) / E_gen)^2) =', np.sqrt(np.mean(np.square((pred_regression - truth_regression) / truth_regression))))

    truth_classification = np.squeeze(truth['classification'])
    print('accuracy', np.mean(np.asarray(np.asarray(pred_classification > 0.5, dtype=np.int32) == truth_classification, dtype=np.float32)))

def make_root_out_entries(inputs, prediction, truth):
    pred_regression, pred_classification = map(np.squeeze, prediction)
    truth_regression = np.squeeze(truth['regression']) * 1.e-2
    truth_classification = np.squeeze(truth['classification'])

    return tuple(inputs) + (pred_regression, pred_classification, truth_regression, truth_classification)

def make_h5_out_data(prediction, truth):
    pred_regression, pred_classification = map(np.squeeze, prediction)
    truth_regression = np.squeeze(truth['regression']) * 1.e-2
    truth_classification = np.squeeze(truth['classification'])

    return {'pred_regression': pred_regression, 'pred_classification': pred_classification, 'truth_regression': truth_regression, 'truth_classification': truth_classification}

def write_ascii_out(inputs, prediction, truth, in_file, out_file, truth_file):
    pred_regression, pred_classification = map(np.squeeze, prediction)

    for entry in zip(*(tuple(inputs) + (truth['regression'], truth['classification'], pred_regression, pred_classification))):
        x, n, truth_r, truth_c, pred_r, pred_c = entry

        in_file.write(' '.join('%f' % v for v in np.reshape(x, (-1,))))
        in_file.write(' %d\n' % n)
        out_file.write('%f\n' % pred_r)
        out_file.write('%f\n' % pred_c)
        truth_file.write('%f\n' % truth_r)
        truth_file.write('%d\n' % truth_c)
