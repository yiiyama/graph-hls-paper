import numpy as np
import sklearn.metrics
try:
    import tensorflow.keras as keras
except ImportError:
    import keras

from layers.caloGraphNN.caloGraphNN_keras import GarNet

# model parameters
n_class = 2
n_vert_max = 128
n_feat = 4
if n_class == 2:
    prob_shape = ('prob', np.float32)
else:
    prob_shape = ('prob', np.float32, (n_class,))

# training parameters

# exported parameters
initial_lr = 1.e-5
generator_args = {'n_vert_max': n_vert_max, 'y_features': 0, 'y_dtype': np.float}
root_out_dtype = [('x', np.float32, (n_vert_max, n_feat)), ('n', np.int32), prob_shape, ('truth', np.int32)]

def _make_model(quantize):
    n_aggregators = 4
    n_filters = 4
    n_propagate = 4
    
    x = keras.layers.Input(shape=(n_vert_max, n_feat))
    n = keras.layers.Input(shape=(1,), dtype='int32')
    inputs = [x, n]

    v = inputs
    v = GarNet(n_aggregators, n_filters, n_propagate, simplified=True, collapse='mean', input_format='xn', output_activation=None, name='gar_1', quantize_transforms=quantize)(v)

    v = keras.layers.Dense(8, activation='relu')(v)
    if n_class == 2:
        v = keras.layers.Dense(1, activation='sigmoid')(v)
    else:
        v = keras.layers.Dense(1, activation='softmax')(v)
    outputs = v
    
    return keras.Model(inputs=inputs, outputs=outputs)

def make_model():
    return _make_model(False)

def make_loss():
    if n_class == 2:
        return 'binary_crossentropy'
    else:
        return 'categorical_crossentropy'

def evaluate_prediction(prediction, truth):
    if n_class == 2:
        prediction = np.squeeze(prediction)
        print('accuracy', np.mean(np.asarray(np.asarray(prediction > 0.5, dtype=np.int32) == truth, dtype=np.float32)))
        #print('bce', sklearn.metrics.log_loss(truth, prediction))
        total = 0.
        batch_size = 64
        for i in range(0, prediction.shape[0], batch_size):
            size = min(batch_size, prediction[i:i + batch_size].shape[0])
            total += sklearn.metrics.log_loss(truth[i:i + batch_size], prediction[i:i + batch_size], eps=np.finfo(np.float32).eps) * size

        print('bce', total / prediction.shape[0])
            
    else:
        truth = np.argmax(truth, axis=1)
        print('accuracy', np.mean(np.asarray(np.argmax(prediction, axis=1) == truth, dtype=np.float32)))

def make_root_out_entries(inputs, prediction, truth):
    if n_class == 2:
        prediction = np.squeeze(prediction)
    else:
        truth = np.argmax(truth, axis=1)

    return tuple(inputs) + (prediction, truth)

def make_h5_out_data(prediction, truth):
    if n_class == 2:
        prediction = np.squeeze(prediction)
    else:
        truth = np.argmax(truth, axis=1)

    return {'prediction': prediction, 'truth': truth}

def write_ascii_out(inputs, prediction, in_file, out_file):
    if n_class == 2:
        prediction = np.squeeze(prediction)

    for xval, nval, pval in zip(*(tuple(inputs) + (prediction,))):
        in_file.write(' '.join('%f' % v for v in np.reshape(xval, (-1,))))
        in_file.write(' %d\n' % nval)
        out_file.write('%f\n' % pval)
