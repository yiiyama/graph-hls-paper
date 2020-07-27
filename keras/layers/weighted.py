from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
try:
    import tensorflow.keras as keras
except ImportError:
    import keras
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from debug_flag import DEBUG

K = keras.backend

NL = 50

class ZWeightedSum(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ZWeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ZWeightedSum, self).build(input_shape)

        self.weight = self.add_weight(name='weights', shape=(NL, 1), initializer=keras.initializers.get('glorot_uniform'), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(NL,), initializer=keras.initializers.get('zeros'), trainable=True)

        layer_low = np.concatenate((np.arange(0., 25., 1., dtype=np.float), np.arange(25., 200., 7., dtype=np.float))) / 200.
        layer_high = np.concatenate((np.arange(1., 26., 1., dtype=np.float), np.arange(32., 207., 7., dtype=np.float))) / 200.

        # bug in input
        layer_low *= 120. / 200.
        layer_high *= 120. / 200.

        self.layer_low = K.constant(layer_low)
        self.layer_high = K.constant(layer_high)

        self.built = True

    def call(self, x):
        data = K.tile(K.expand_dims(x, axis=1), (1, NL, 1, 1))
        zero = K.zeros_like(data[:, :, :, 0])

        zcoord = K.permute_dimensions(data[:, :, :, 0], (0, 2, 1))
        condition = K.all(K.stack((K.greater(zcoord, self.layer_low), K.less(zcoord, self.layer_high)), axis=-1), axis=-1)
        condition = K.permute_dimensions(condition, (0, 2, 1))
        histogram = K.sum(K.switch(condition, data[:, :, :, 1], zero), axis=2)

        histogram = nn.bias_add(histogram, self.bias)

        output = K.squeeze(gen_math_ops.mat_mul(histogram, self.weight), axis=-1)

        return output

    def compute_output_shape(self, input_shape):
        return (1,)

    def get_config(self):
        config = super(ZWeightedSum, self).get_config()

        return config
