import tensorflow as tf

def instance_norm(inputs, scope):
    with tf.variable_scope(scope):
        filters = inputs.get_shape()[3]
        scale = tf.get_variable('scale', [filters], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable('offset', [filters], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (inputs - mean) * inv

        return scale * normalized + offset


def relu(inputs):
    return tf.nn.relu(inputs)


def tanh(inputs):
    return tf.nn.tanh(inputs)


def lrelu(inputs, alpha=0.01):
    return tf.nn.leaky_relu(inputs, alpha=alpha)


def conv2d(inputs, filters, kernel_size, strides, padding='SAME', scope='conv_2d'):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [kernel_size, kernel_size, inputs.get_shape()[-1], filters],
                            initializer=tf.random_normal_initializer(stddev=0.02))

        conv = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding=padding)

    return conv


def deconv2d(inputs, filters, kernel_size, strides, padding='SAME', scope='deconv_2d'):
    with tf.variable_scope(scope):

        deconv = tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides, padding)

    return deconv


def dense(inputs, units, scope='dense'):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [inputs.get_shape().as_list()[-1], units],
                            initializer=tf.random_normal_initializer(stddev=0.02))

        dense = tf.matmul(inputs, w)
    return dense


