import keras
from layers.simple import GarNet

# model parameters
n_class = 2
n_vert_max = 256
n_feat = 4

# training parameters

# exported parameters
initial_lr = 0.00005
generator_args = {'n_vert_max': n_vert_max}

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
