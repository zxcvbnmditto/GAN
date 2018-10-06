import tensorflow as tf
import numpy as np
import warnings

NO_OPS = 'NO_OPS'


def scope_has_variables(scope):
    return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)) > 0

# from https://github.com/ctwxdd/Tensorflow-ACGAN-Anime-Generation
def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

# from https://github.com/ctwxdd/Tensorflow-ACGAN-Anime-Generation
# def spectral_norm(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
#     # Usually num_iters = 1 will be enough
#     W_shape = W.shape.as_list()
#     W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
#
#     if u is None:
#         u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
#
#     def power_iteration(i, u_i, v_i):
#         v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
#         u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
#         return i + 1, u_ip1, v_ip1
#
#     _, u_final, v_final = tf.while_loop(
#         cond=lambda i, _1, _2: i < num_iters,
#         body=power_iteration,
#         loop_vars=(tf.constant(0, dtype=tf.int32),
#                    u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
#     )
#
#     if update_collection is None:
#         warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
#                       '. Please consider using a update collection instead.')
#         sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
#         # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
#         W_bar = W_reshaped / sigma
#
#         with tf.control_dependencies([u.assign(u_final)]):
#             W_bar = tf.reshape(W_bar, W_shape)
#     else:
#         sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
#         # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
#         W_bar = W_reshaped / sigma
#         W_bar = tf.reshape(W_bar, W_shape)
#         # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
#         # has already been collected on the first call.
#         if update_collection != NO_OPS:
#             tf.add_to_collection(update_collection, u.assign(u_final))
#
#     if with_sigma:
#         return W_bar, sigma
#     else:
#         return W_bar

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
       power iteration
       Usually iteration = 1 will be enough
       """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def batch_norm(inputs, training):
    return tf.layers.batch_normalization(inputs=inputs, momentum=0.9, epsilon=1e-5, training=training, name="bn")


def layer_norm(inputs):
    return tf.contrib.layers.layer_norm(inputs=inputs, scope="ln")


def conv_2d(inputs, filters, kernel_size, strides, padding, stddev, training, norm, collection=None, scope='conv_2d'):
    in_dim = kernel_size * kernel_size * inputs.get_shape().as_list()[-1]

    if stddev is None:
        stddev = np.sqrt(2. / (in_dim))

    with tf.variable_scope(scope):
        if scope_has_variables(scope):
            print("test")
            scope.reuse_variables()

        w = tf.get_variable("w", [kernel_size, kernel_size, inputs.get_shape()[-1], filters],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        # b = tf.get_variable("b", [filters], initializer=tf.constant_initializer(0.0))

        if norm is "spectral_norm":
            # w = spectral_norm(w, update_collection=collection)
            w = spectral_norm(w)


        conv = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding=padding)
        # conv = tf.nn.bias_add(conv, b)

        if norm is "batch_norm":
            conv = batch_norm(conv, training)
        elif norm is "layer_norm":
            conv = layer_norm(conv)

    return conv


def dense(inputs, units, stddev, training, norm, collection=None, scope='dense'):
    in_dim = inputs.get_shape().as_list()[-1]

    if stddev is None:
        stddev = np.sqrt(2. / (in_dim))

    with tf.variable_scope(scope):
        if scope_has_variables(scope):
            print("test")
            scope.reuse_variables()

        w = tf.get_variable("w", [inputs.get_shape().as_list()[-1], units],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        # b = tf.get_variable("b", [units], initializer=tf.constant_initializer(0.0))

        if norm is "spectral_norm":
            # w = spectral_norm(w, update_collection=collection)
            w = spectral_norm(w)


        dense = tf.matmul(inputs, w)
        # dense = tf.nn.bias_add(dense, b)

        if norm is "batch_norm":
            dense = batch_norm(dense, training)
        elif norm is "layer_norm":
            dense = layer_norm(dense)

    return dense

# from https://github.com/ctwxdd/Tensorflow-ACGAN-Anime-Generation
def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

# from https://github.com/ctwxdd/Tensorflow-ACGAN-Anime-Generation
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output
