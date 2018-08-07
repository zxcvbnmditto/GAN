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

            # 768 - 384 - 192 - 96 - 3 => 6 * 6 * 768 - 1024 - 1
            #   6 -  12 -  24 - 48 - 96

            # Conv layer 1
            w1 = tf.get_variable('d_w1', [4, 4, 3, 96], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('d_b1', [96], initializer=tf.constant_initializer(0))
            conv_1 = tf.nn.leaky_relu(self.conv2d(x, w1) + b1)
            pool_1 = self.pooling(conv_1)

            # Conv layer 2
            w2 = tf.get_variable('d_w2', [4, 4, 96, 192], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('d_b2', [192], initializer=tf.constant_initializer(0))
            conv_2 = tf.nn.leaky_relu(self.conv2d(pool_1, w2) + b2)
            pool_2 = self.pooling(conv_2)

            # Conv layer 3
            w3 = tf.get_variable('d_w3', [4, 4, 192, 384], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b3 = tf.get_variable('d_b3', [384], initializer=tf.constant_initializer(0))
            conv_3 = tf.nn.leaky_relu(self.conv2d(pool_2, w3) + b3)
            pool_3 = self.pooling(conv_3)

            # Conv layer 4
            w4 = tf.get_variable('d_w4', [4, 4, 384, 768], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b4 = tf.get_variable('d_b4', [768], initializer=tf.constant_initializer(0))
            conv_4 = tf.nn.leaky_relu(self.conv2d(pool_3, w4) + b4)
            pool_4 = self.pooling(conv_4)

            flatten = tf.layers.flatten(pool_4)

            # Dense layer 1
            w5 = tf.get_variable('d_w5', [6 * 6 * 768, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b5 = tf.get_variable('d_b5', [1024], initializer=tf.constant_initializer(0))
            dense1 = tf.matmul(flatten, w5) + b5
            dense1 = tf.nn.leaky_relu(dense1)

            # Dense layer 2
            w6 = tf.get_variable('d_w6', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b6 = tf.get_variable('d_b6', [1], initializer=tf.constant_initializer(0))
            dense2 = tf.matmul(dense1, w6) + b6

            outputs = dense2

        return outputs

    def generator(self, x, reuse=False):
        print("Generator ...........")
        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            # Dense layer 1
            w1 = tf.get_variable('g_w1', [100, 768 * 6 * 6], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('g_b1', [1], initializer=tf.constant_initializer(0))

            dense1 = tf.nn.relu(tf.matmul(x, w1) + b1)
            dense1 = tf.reshape(dense1, [self.batch_size, 6, 6, 768])

            # 768 - 384 - 192 - 96 - 3
            #   6 -  12 -  24 - 48 - 96

            # Deconv layer 1
            o_shape1 = [self.batch_size, 12, 12, 384]
            w2 = tf.get_variable('g_w2', [4, 4, 384, 768], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b2 = tf.get_variable('g_b2', [384], initializer=tf.constant_initializer(0.1))
            deconv1 = self.deconv2d(dense1, w2, o_shape1) + b2
            deconv1 = tf.contrib.layers.batch_norm(inputs=deconv1, decay=0.9, is_training=True,
                                                   scope="g_bn1")

            deconv1 = tf.nn.relu(deconv1)

            # Deconv layer 2
            o_shape2 = [self.batch_size, 24, 24, 192]
            w3 = tf.get_variable('g_w3', [4, 4, 192, 384], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b3 = tf.get_variable('g_b3', [192], initializer=tf.constant_initializer(0.1))
            deconv2 = self.deconv2d(deconv1, w3, o_shape2) + b3
            deconv2 = tf.contrib.layers.batch_norm(inputs=deconv2, decay=0.9, is_training=True,
                                                   scope="g_bn2")

            deconv2 = tf.nn.relu(deconv2)

            # Deconv layer 3
            o_shape3 = [self.batch_size, 48, 48, 96]
            w4 = tf.get_variable('g_w4', [4, 4, 96, 192], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b4 = tf.get_variable('g_b4', [96], initializer=tf.constant_initializer(0.1))
            deconv3 = self.deconv2d(deconv2, w4, o_shape3) + b4
            deconv3 = tf.contrib.layers.batch_norm(inputs=deconv3, decay=0.9, is_training=True,
                                                   scope="g_bn3")

            deconv3 = tf.nn.relu(deconv3)

            # Deconv layer 4
            o_shape4 = [self.batch_size, 96, 96, 3]
            w5 = tf.get_variable('g_w5', [4, 4, 3, 96], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b5 = tf.get_variable('g_b5', [3], initializer=tf.constant_initializer(0.1))
            deconv4 = self.deconv2d(deconv3, w5, o_shape4) + b5

            outputs = tf.nn.tanh(deconv4)

        return outputs

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
        self.d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DG, labels=tf.zeros_like(
            self.DG)))

        self.d_loss = self.d_real + self.d_fake

        # Summary doesn't work bc placeholder is fed in later than build()
        # d_loss = tf.summary.scalar('d_loss', self.d_loss)
        # g_loss = tf.summary.scalar('g_loss', self.g_loss)
        #
        # self.summary_op = tf.summary.merge([d_loss, g_loss])

        d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]

        # print("D_vars ........", d_vars)
        # print("G_vars ........", g_vars)

        with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
            print("reuse or not: {}".format(tf.get_variable_scope().reuse))
            assert tf.get_variable_scope().reuse == False, "Houston tengo un problem"

            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999).minimize(
                self.d_loss, var_list=d_vars)

            self.trainerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999).minimize(
                self.g_loss, var_list=g_vars)

        self.saver = tf.train.Saver(tf.global_variables())

