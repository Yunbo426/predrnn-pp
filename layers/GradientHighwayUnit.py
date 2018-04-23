__author__ = 'yunbo'

import tensorflow as tf
from layers.TensorLayerNorm import tensor_layer_norm

class GHU():
    def __init__(self, layer_name, filter_size, num_features, tln=False,
                 initializer=0.001):
        """Initialize the Gradient Highway Unit.
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_features = num_features
        self.layer_norm = tln
        if initializer == -1:
            self.initializer = None
        else:
            self.initializer = tf.random_uniform_initializer(-initializer,initializer)

    def init_state(self, inputs, num_features):
        dims = inputs.get_shape().ndims
        if dims == 4:
            batch = inputs.get_shape()[0]
            height = inputs.get_shape()[1]
            width = inputs.get_shape()[2]
        else:
            raise ValueError('input tensor should be rank 4.')
        return tf.zeros([batch, height, width, num_features], dtype=tf.float32)

    def __call__(self, x, z):
        if z is None:
            z = self.init_state(x, self.num_features)
        with tf.variable_scope(self.layer_name):
            z_concat = tf.layers.conv2d(
                z, self.num_features*2,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='state_to_state')
            if self.layer_norm:
                z_concat = tensor_layer_norm(z_concat, 'state_to_state')

            x_concat = tf.layers.conv2d(
                x, self.num_features*2,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='input_to_state')
            if self.layer_norm:
                x_concat = tensor_layer_norm(x_concat, 'input_to_state')

            gates = tf.add(x_concat, z_concat)
            p, u = tf.split(gates, 2, 3)
            p = tf.nn.tanh(p)
            u = tf.nn.sigmoid(u)
            z_new = u * p + (1-u) * z
            return z_new

