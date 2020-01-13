import keras
from layers.simple import GarNet

n_vert = 256
n_feat = 6
n_aggregators = 4
n_filters = 4
n_propagate = 4
n_class = 2

x = keras.layers.Input(shape=(n_vert, n_feat))
n = keras.layers.Input(shape=(1,), dtype='int32')
inputs = [x, n]

v = inputs
v = GarNet(n_aggregators, n_filters, n_propagate, collapse='mean', deduce_nvert=False, name='gar_1')(v)
v = keras.layers.Dense(1, activation='sigmoid')(v)
outputs = v

model = keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    import sys

    out_path = sys.argv[1]

    with open(out_path, 'w') as json_file:
        json_file.write(model.to_json())
