#Reference
#An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution

import tensorflow as tf

class CoordConv2D:
    def __init__(self, k_size, filters, 
                 strides=1, padding='same',
                 with_r=False, activation=None,
                 kernel_initializer=None, name=None):

        self.with_r = with_r

        self.conv_kwargs = {
            'filters': filters,
            'kernel_size': k_size,
            'strides': strides,
            'padding': padding,
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'name': name,
        }

    def __call__(self, in_tensor):
        with tf.name_scope('coord_conv'):
            batch_size = tf.shape(in_tensor)[0]
            x_dim = tf.shape(in_tensor)[1]
            y_dim  = tf.shape(in_tensor)[2]

            #example (32, 256, 256, 3)
            #add 0 ~ 255 feature map to (32, 256, 256, 1)
            xx_indices = tf.tile(
                tf.expand_dims(tf.expand_dims(tf.range(x_dim), 0), 0),
                [batch_size, y_dim, 1])
            xx_indices = tf.expand_dims(xx_indices, -1)

            yy_indices = tf.tile(
                tf.expand_dims(tf.reshape(tf.range(y_dim), (y_dim, 1)), 0),
                [batch_size, 1, x_dim])
            yy_indices = tf.expand_dims(yy_indices, -1)

            #normalize xx_indices from 0~255 to 0~1 (32, 256, 256, 1)
            xx_indices = tf.divide(xx_indices, x_dim - 1)
            yy_indices = tf.divide(yy_indices, y_dim - 1)

            #0~1 * 2 = 0~2
            #0~2 - 1 = -1~1
            xx_indices = tf.cast(tf.subtract(tf.multiply(xx_indices, 2.), 1.),
                                 dtype=in_tensor.dtype)
            yy_indices = tf.cast(tf.subtract(tf.multiply(yy_indices, 2.), 1.),
                                 dtype=in_tensor.dtype)

            #tf concat original, x_axis and y_axis maps
            processed_tensor = tf.concat([in_tensor, xx_indices, yy_indices], axis=-1)

            #-1~1 - 0.5 = -1.5~0.5
            #(-1.5~0.5) ^ 2 = 2.25~0.25
            if self.with_r:
                rr = tf.sqrt(tf.add(tf.square(xx_indices - 0.5), tf.square(yy_indices - 0.5)))
                processed_tensor = tf.concat([processed_tensor, rr], axis=-1)

            return tf.keras.layers.Conv2D(**self.conv_kwargs)(processed_tensor)
		
