import keras
import keras.backend as K

class GarNet(keras.layers.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate, deduce_nvert=False, **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self.n_aggregators = n_aggregators
        self.n_filters = n_filters
        self.n_propagate = n_propagate

        self.input_feature_transform = keras.layers.Dense(n_propagate, name=self.name+'_FLR')
        self.aggregator_distance = keras.layers.Dense(n_aggregators, name=self.name+'_S')
        self.output_feature_transform = keras.layers.Dense(n_filters, activation='tanh', name=self.name+'_Fout')

        self._sublayers = [self.input_feature_transform, self.aggregator_distance, self.output_feature_transform]

    def build(self, input_shape):
        self.input_feature_transform.build(input_shape)
        self.aggregator_distance.build(input_shape)
        
        # tf.ragged FIXME? tf.shape()?
        self.output_feature_transform.build((input_shape[0], input_shape[1], input_shape[2] + self.aggregator_distance.units + 2 * self.aggregator_distance.units * (self.input_feature_transform.units + self.aggregator_distance.units)))

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

        super(GarNet, self).build(input_shape)

    def call(self, x):
        features = self.input_feature_transform(x) # (B, V, F)
        distance = self.aggregator_distance(x) # (B, V, S)

        edge_weights = K.exp(-1. * K.square(distance))

        features = K.concatenate([features, edge_weights], axis=-1) # (B, V, F+S)

        # vertices -> aggregators
        edge_weights_trans = K.permute_dimensions(edge_weights, (0, 2, 1)) # (B, S, V)
        aggregated_max = self.apply_edge_weights(features, edge_weights_trans, aggregation=K.max) # (B, S, F+S)
        aggregated_mean = self.apply_edge_weights(features, edge_weights_trans, aggregation=K.mean) # (B, S, F+S)

        aggregated = K.concatenate([aggregated_max, aggregated_mean], axis=-1) # (B, S, 2*(F+S))

        # aggregators -> vertices
        updated_features = self.apply_edge_weights(aggregated, edge_weights) # (B, V, 2*S*(F+S))

        updated_features = K.concatenate([x, updated_features, edge_weights], axis=-1) # (B, V, X+2*S*(F+S)+S)

        return self.output_feature_transform(updated_features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_filters)

    def apply_edge_weights(self, features, edge_weights, aggregation=None):
        features = K.expand_dims(features, axis=1) # (B, 1, v, f)
        edge_weights = K.expand_dims(edge_weights, axis=3) # (B, A, v, 1)

        # tf.ragged FIXME? broadcasting should work
        out = edge_weights * features # (B, u, v, f)
        # tf.ragged FIXME? these values won't work
        n = features.shape[-2].value * features.shape[-1].value

        if aggregation:
            out = aggregation(out, axis=2) # (B, u, f)
            n = features.shape[-1].value
        
        # tf.ragged FIXME? there might be a chance to spell out batch dim instead and use -1 for vertices?
        return K.reshape(out, [-1, out.shape[1].value, n]) # (B, u, n)
    
    def get_config(self):
        config = super(GarNet, self).get_config()
        config.update({'n_aggregators': self.n_aggregators, 'n_filters': self.n_filters, 'n_propagate': self.n_propagate})
        return config
