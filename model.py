import tensorflow as tf


class Vanilla_Gan():
    def __init__(self, learning_rate, batch_size):
        print("Constructing VANILLA GAN model ........")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.X = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
        self.Z = tf.placeholder(tf.float32, shape=[None, 100])

        self.build()

    # filter shape = [filter_width, filter_height, channel (prev filter number), output_filter_num]
    # stride shape = [1, width_stride, height_stride, 1]
    def conv2d(self, x, w):
        return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME')

    def deconv2d(self, x, w, shape):
        return tf.nn.conv2d_transpose(value=x, filter=w, output_shape=shape, strides=[1, 2, 2, 1], padding='SAME')

    # ksize => size of the window to pool
    def pooling(self, x):
        return tf.nn.avg_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def discriminator(self, x, reuse=False):
        print("Discriminator ...........")
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            # Conv layer 1
            w1 = tf.get_variable('d_w1', [4, 4, 3, 24], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('d_b1', [24], initializer=tf.constant_initializer(0))
            conv_1 = tf.nn.relu(self.conv2d(x, w1) + b1)
            pool_1 = self.pooling(conv_1)

            # Conv layer 2
            w2 = tf.get_variable('d_w2', [4, 4, 24, 48], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('d_b2', [48], initializer=tf.constant_initializer(0))
            conv_2 = tf.nn.relu(self.conv2d(pool_1, w2) + b2)
            pool_2 = self.pooling(conv_2)

            # Conv layer 3
            w3 = tf.get_variable('d_w3', [4, 4, 48, 96], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b3 = tf.get_variable('d_b3', [96], initializer=tf.constant_initializer(0))
            conv_3 = tf.nn.relu(self.conv2d(pool_2, w3) + b3)
            pool_3 = self.pooling(conv_3)

            # Conv layer 4
            w4 = tf.get_variable('d_w4', [4, 4, 96, 192], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b4 = tf.get_variable('d_b4', [192], initializer=tf.constant_initializer(0))
            conv_4 = tf.nn.relu(self.conv2d(pool_3, w4) + b4)
            pool_4 = self.pooling(conv_4)

            flatten = tf.layers.flatten(pool_4)

            w5 = tf.get_variable('d_w5', [6 * 6 * 192, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b5 = tf.get_variable('d_b5', [1], initializer=tf.constant_initializer(0))

            outputs = tf.nn.sigmoid(tf.matmul(flatten, w5) + b5)

        return outputs

    def generator(self, x, reuse=False):
        print("Generator ...........")
        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            # Dense layer 1
            w1 = tf.get_variable('g_w1', [100, 192 * 24 * 24], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('g_b1', [1], initializer=tf.constant_initializer(0))

            dense1 = tf.nn.relu(tf.matmul(x, w1) + b1)
            dense1 = tf.reshape(dense1, [self.batch_size, 24, 24, 192])

            # 192 -> 96 -> 3

            # Deconv layer 1
            o_shape1 = [self.batch_size, 48, 48, 96]
            w2 = tf.get_variable('g_w2', [4, 4, 96, 192], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b2 = tf.get_variable('g_b2', [96], initializer=tf.constant_initializer(0.1))
            deconv1 = self.deconv2d(dense1, w2, o_shape1) + b2
            deconv1 = tf.contrib.layers.batch_norm(inputs=deconv1, center=True, scale=True, is_training=True,
                                                   scope="g_bn1")

            deconv1 = tf.nn.relu(deconv1)

            # Deconv layer 2
            o_shape2 = [self.batch_size, 96, 96, 3]
            w3 = tf.get_variable('g_w3', [4, 4, 3, 96], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b3 = tf.get_variable('g_b3', [3], initializer=tf.constant_initializer(0.1))
            deconv2 = self.deconv2d(deconv1, w3, o_shape2) + b3
            deconv2 = tf.contrib.layers.batch_norm(inputs=deconv2, center=True, scale=True, is_training=True,
                                                   scope="g_bn2")

            deconv2 = tf.nn.relu(deconv2)

        return deconv2

    def build(self):
        # real => images from database
        self.Dx = self.discriminator(self.X, reuse=False)
        self.Gz = self.generator(self.Z, reuse=False)
        # fake => images generated by gaussian distribution
        self.DG = self.discriminator(self.Gz, reuse=True)

        # generator loss => back propagate of the quality of fake images
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DG, labels=tf.ones_like(
            self.DG)))

        # discriminator => back propagate of the combined performance of distinguishing real and fake images
        self.d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx, labels=tf.ones_like(
            self.Dx)))
        self.d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DG, labels=tf.ones_like(
            self.DG)))

        self.d_loss = self.d_real + self.d_fake

        # Summary doesn't work bc placeholder is fed in later than build()
        # d_loss = tf.summary.scalar('d_loss', self.d_loss)
        # g_loss = tf.summary.scalar('g_loss', self.g_loss)
        #
        # self.summary_op = tf.summary.merge([d_loss, g_loss])

        d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]

        with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
            print("reuse or not: {}".format(tf.get_variable_scope().reuse))
            assert tf.get_variable_scope().reuse == False, "Houston tengo un problem"
            self.trainerD = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=d_vars)
            self.trainerG = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=g_vars)
        print("exiting")

        self.saver = tf.train.Saver(tf.global_variables())
