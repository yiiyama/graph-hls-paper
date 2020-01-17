import keras
from layers.simple import GarNet

def make_model(n_vert, n_feat, n_class=2):
    n_aggregators = 4
    n_filters = 4
    n_propagate = 4
    
    x = keras.layers.Input(shape=(n_vert, n_feat))
    n = keras.layers.Input(shape=(1,), dtype='int32')
    inputs = [x, n]
    
    v = inputs
    v = GarNet(n_aggregators, n_filters, n_propagate, collapse='mean', deduce_nvert=False, name='gar_1')(v)
    if n_class == 2:
        v = keras.layers.Dense(1, activation='sigmoid')(v)
    else:
        v = keras.layers.Dense(1, activation='softmax')(v)
    outputs = v
    
    return keras.Model(inputs=inputs, outputs=outputs)

def make_loss(n_class=2):
    if n_class == 2:
        return 'binary_crossentropy'
    else:
        return 'categorical_crossentropy'


if __name__ == '__main__':
    import sys

    out_path = sys.argv[1]
    n_feat = int(sys.argv[2])

    model = make_model(list(range(n_feat)), n_class=2)

    with open(out_path, 'w') as json_file:
        json_file.write(model.to_json())
