import keras
from layers.caloGraphNN.caloGraphNN_keras import GarNet

# model parameters
n_class = 2
n_vert_max = 256
n_feat = 4
if n_class == 2:
    prob_shape = ('prob', np.float32)
else:
    prob_shape = ('prob', np.float32, (n_class,))

# training parameters

# exported parameters
initial_lr = 0.00005
generator_args = {'n_vert_max': n_vert_max}
root_out_dtype = [('x', np.float32, (n_vert_max, n_feat)), ('n', np.int32), prob_shape, ('truth', np.int32)]

def make_model():
    n_aggregators = 4
    n_filters = 4
    n_propagate = 4
    
    x = keras.layers.Input(shape=(n_vert_max, n_feat))
    n = keras.layers.Input(shape=(1,), dtype='int32')
    inputs = [x, n]
    
    v = inputs
    v = GarNet(n_aggregators, n_filters, n_propagate, collapse='mean', input_format='xn', name='gar_1')(v)
    if n_class == 2:
        v = keras.layers.Dense(1, activation='sigmoid')(v)
    else:
        v = keras.layers.Dense(1, activation='softmax')(v)
    outputs = v
    
    return keras.Model(inputs=inputs, outputs=outputs)

def make_loss():
    if n_class == 2:
        return 'binary_crossentropy'
    else:
        return 'categorical_crossentropy'

def evaluate_prediction(prediction, truth):
    if n_class == 2:
        prediction = np.squeeze(prediction)
        print('accuracy', np.mean(np.asarray(np.asarray(prediction > 0.5, dtype=np.int32) == truth, dtype=np.float32)))
    else:
        truth = np.argmax(truth, axis=1)
        print('accuracy', np.mean(np.asarray(np.argmax(prediction, axis=1) == truth, dtype=np.float32)))

def make_root_out_entries(inputs, prediction, truth):
    if n_class == 2:
        prediction = np.squeeze(prediction)
    else:
        truth = np.argmax(truth, axis=1)

    return tuple(inputs) + (prediction, truth)

def write_ascii_out(inputs, prediction, in_file, out_file):
    if n_class == 2:
        prediction = np.squeeze(prediction)

    for xval, nval, pval in zip(*(tuple(inputs) + (prediction,))):
        in_file.write(' '.join('%f' % v for v in np.reshape(xval, (-1,))))
        in_file.write(' %d\n' % nval)
        out_file.write('%f\n' % pval)
