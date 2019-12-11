import math
import keras
import keras.backend as K
import numpy as np

DEBUG = False

class GarNet(keras.layers.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate, collapse=None, deduce_nvert=False, discretize_distance=False, **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self.n_aggregators = n_aggregators
        self.n_filters = n_filters
        self.n_propagate = n_propagate

        if collapse is None:
            self.collapse = None
        elif collapse in ['mean', 'sum', 'max']:
            self.collapse = collapse
        else:
            raise NotImplementedError('Unsupported collapse operation')

        self.deduce_nvert = deduce_nvert
        self.discretize_distance = discretize_distance

        self.input_feature_transform = keras.layers.Dense(n_propagate, name=self.name+'/FLR')
        self.aggregator_distance = keras.layers.Dense(n_aggregators, name=self.name+'/S')
        self.output_feature_transform = keras.layers.Dense(n_filters, name=self.name+'/Fout')

        self._sublayers = [self.input_feature_transform, self.aggregator_distance, self.output_feature_transform]

    def build(self, input_shape):
        super(GarNet, self).build(input_shape)

        if self.deduce_nvert:
            data_shape = input_shape
        else:
            data_shape, _ = input_shape

        self.input_feature_transform.build(data_shape)
        self.aggregator_distance.build(data_shape)
        self.output_feature_transform.build(data_shape[:2] + (self.n_aggregators * self.n_propagate,))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

    def call(self, x):
        if self.deduce_nvert:
            data = x

            vertex_mask = K.cast(K.not_equal(data[..., 3:4], 0.), 'float32')

        else:
            data, num_vertex = x
    
            if DEBUG:
                data = K.print_tensor(data, message='data is ', summarize=-1)
                num_vertex = K.print_tensor(num_vertex, message='num_vertex is ', summarize=-1)
    
            data_shape = K.shape(data)
            B = data_shape[0]
            V = data_shape[1]
            vertex_indices = K.tile(K.expand_dims(K.arange(0, V), axis=0), (B, 1)) # (B, [0..V-1])
            num_vertex = K.cast(num_vertex, 'int32')
            vertex_mask = K.expand_dims(K.cast(K.less(vertex_indices, num_vertex), 'float32'), axis=-1) # (B, V, 1)

        if DEBUG:
            vertex_mask = K.print_tensor(vertex_mask, message='vertex_mask is ', summarize=-1)

        features = self.input_feature_transform(data) # (B, V, F)
        distance = self.aggregator_distance(data) # (B, V, S)

        if DEBUG:
            features = K.print_tensor(features, message='features is ', summarize=-1)
            distance = K.print_tensor(distance, message='distance is ', summarize=-1)

        if self.discretize_distance:
            distance = K.round(distance)
            if DEBUG:
                distance = K.print_tensor(distance, message='rounded distance is ', summarize=-1)

        edge_weights = vertex_mask * K.exp(distance * (-math.log(2.))) # (B, V, S)
        
        if DEBUG:
            edge_weights = K.print_tensor(edge_weights, message='edge_weights is ', summarize=-1)

        # vertices -> aggregators
        edge_weights_trans = K.permute_dimensions(edge_weights, (0, 2, 1)) # (B, S, V)
        aggregated = self._apply_edge_weights(features, edge_weights_trans, aggregation=K.mean) # (B, S, F)

        if DEBUG:
            aggregated = K.print_tensor(aggregated, message='aggregated is ', summarize=-1)

        # aggregators -> vertices
        updated_features = self._apply_edge_weights(aggregated, edge_weights) # (B, V, S*F)

        if DEBUG:
            updated_features = K.print_tensor(updated_features, message='updated_features is ', summarize=-1)

        output = vertex_mask * self.output_feature_transform(updated_features)

        if self.collapse == 'mean':
            output = K.mean(output, axis=1)
        elif self.collapse == 'sum':
            output = K.sum(output, axis=1)
        elif self.collapse == 'max':
            output = K.max(output, axis=1)

        if DEBUG:
            output = K.print_tensor(output, message='output is ', summarize=-1)

        return output

    def compute_output_shape(self, input_shape):
        if self.deduce_nvert:
            data_shape = input_shape
        else:
            data_shape, _ = input_shape

        if self.collapse is None:
            return data_shape[0:2] + (self.n_filters,)
        else:
            return (data_shape[0], self.n_filters)

    def get_config(self):
        config = super(GarNet, self).get_config()
        config.update({
            'n_aggregators': self.n_aggregators,
            'n_filters': self.n_filters,
            'n_propagate': self.n_propagate,
            'collapse': self.collapse,
            'deduce_nvert': self.deduce_nvert,
            'discretize_distance': self.discretize_distance
        })

        return config

    def _apply_edge_weights(self, features, edge_weights, aggregation=None):
        features = K.expand_dims(features, axis=1) # (B, 1, v, f)
        edge_weights = K.expand_dims(edge_weights, axis=3) # (B, u, v, 1)

        if DEBUG:
            features = K.print_tensor(features, message='applying on features ', summarize=-1)
            edge_weights = K.print_tensor(edge_weights, message='applying weights ', summarize=-1)

        out = edge_weights * features # (B, u, v, f)

        if DEBUG:
            out = K.print_tensor(out, message='before aggregation ', summarize=-1)

        if aggregation:
            out = aggregation(out, axis=2) # (B, u, f)
        else:
            out = K.reshape(out, (-1, edge_weights.shape[1].value, features.shape[-1].value * features.shape[-2].value))
        
        return out
