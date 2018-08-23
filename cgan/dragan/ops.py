import tensorflow as tf

def g_res_block(inputs, filters, kernel_size, stride, training=True):
    # Conv
    outputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=stride, padding='same')
    # Bn
    outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, epsilon=1e-5, scale=True, is_training=training, trainable=True)
    # Relu
    outputs = tf.nn.relu(outputs)
    # Conv
    outputs = tf.layers.conv2d(outputs, filters=filters, kernel_size=kernel_size, strides=stride, padding='same')
    # Bn
    outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, epsilon=1e-5, scale=True, is_training=training, trainable=True)
    # Elementwise
    outputs = outputs + inputs

    return outputs

def d_block_1(inputs, filters, kernel_size=4, stride=2):
    outputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=stride, padding='same')
    outputs = tf.nn.leaky_relu(outputs, alpha=0.2)

    return outputs

def d_block_2(inputs, filters, kernel_size=3, stride=1):
    outputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, strides=stride, padding='same')
    outputs = tf.nn.leaky_relu(outputs, alpha=0.2)
    outputs = tf.layers.conv2d(outputs, filters=filters, kernel_size=kernel_size, strides=stride, padding='same')
    outputs = outputs + inputs
    outputs = tf.nn.leaky_relu(outputs, alpha=0.2)

    return outputs

# from https://github.com/ctwxdd/Tensorflow-ACGAN-Anime-Generation/blob/master/libs/ops.py
def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

# from https://github.com/ctwxdd/Tensorflow-ACGAN-Anime-Generation/blob/master/libs/ops.py
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