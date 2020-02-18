from __future__ import absolute_import, division, print_function, unicode_literals

import math
import keras
import keras.backend as K

from debug_flag import DEBUG
from layers.simple import GarNet

class GarNetStack(GarNet):
    def _setup_transforms(self, n_aggregators, n_filters, n_propagate, output_activation):
        self.transform_layers = []
        # inputs are lists
        for it, (p, a, f) in enumerate(zip(n_propagate, n_aggregators, n_filters)):
            self.transform_layers.append((
                keras.layers.Dense(p, name=(self.name + ('/FLR%d' % it))),
                keras.layers.Dense(a, name=(self.name + ('/S%d' % it))),
                keras.layers.Dense(f, activation=output_activation, name=(self.name + ('/Fout%d' % it)))
            ))

        self._sublayers = sum((list(layers) for layers in self.transform_layers), [])

    def _build_transforms(self, data_shape):
        for in_transform, d_compute, out_transform in self.transform_layers:
            in_transform.build(data_shape)
            d_compute.build(data_shape)
            out_transform.build(data_shape[:2] + (d_compute.units * in_transform.units,))

            data_shape = data_shape[:2] + (out_transform.units,)

    def call(self, x):
        data, num_vertex, vertex_mask = self._unpack_input(x)

        for in_transform, d_compute, out_transform in self.transform_layers:
            data = self._garnet(data, num_vertex, vertex_mask, in_transform, d_compute, out_transform)
    
        output = self._collapse_output(data)

        return output

    def compute_output_shape(self, input_shape):
        return self._get_output_shape(input_shape, self.transform_layers[-1][2])

    def _add_transform_config(self, config):
        config.update({
            'n_propagate': list(ll[0].units for ll in self.transform_layers),
            'n_aggregators': list(ll[1].units for ll in self.transform_layers),
            'n_filters': list(ll[2].units for ll in self.transform_layers),
            'n_sublayers': len(self.transform_layers)
        })
