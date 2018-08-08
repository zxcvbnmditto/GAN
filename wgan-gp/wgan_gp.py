import tensorflow as tf

class WGAN_GP():
    def __init__(self, learning_rate, batch_size, latent_size):
        print("Constructing WGAN model ........")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.Lambda = 10
        self.X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.latent_size])

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

            # 256 - 128 - 64 - 32 - 3
            #  64 - 32  - 16 -  8 - 4

            # Conv layer 1
            w1 = tf.get_variable('d_w1', [4, 4, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
            conv_1 = tf.nn.leaky_relu(self.conv2d(x, w1) + b1)
            pool_1 = self.pooling(conv_1)

            # Conv layer 2
            w2 = tf.get_variable('d_w2', [4, 4, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
            conv_2 = tf.nn.leaky_relu(self.conv2d(pool_1, w2) + b2)
            pool_2 = self.pooling(conv_2)

            # Conv layer 3
            w3 = tf.get_variable('d_w3', [4, 4, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b3 = tf.get_variable('d_b3', [128], initializer=tf.constant_initializer(0))
            conv_3 = tf.nn.leaky_relu(self.conv2d(pool_2, w3) + b3)
            pool_3 = self.pooling(conv_3)

            # Conv layer 4
            w4 = tf.get_variable('d_w4', [4, 4, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b4 = tf.get_variable('d_b4', [256], initializer=tf.constant_initializer(0))
            conv_4 = tf.nn.leaky_relu(self.conv2d(pool_3, w4) + b4)
            pool_4 = self.pooling(conv_4)

            flatten = tf.layers.flatten(pool_4)

            # Dense layer 1
            w5 = tf.get_variable('d_w5', [4 * 4 * 256, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b5 = tf.get_variable('d_b5', [1], initializer=tf.constant_initializer(0))
            dense1 = tf.matmul(flatten, w5) + b5

            outputs = dense1

        return outputs

    def generator(self, x, reuse=False):
        print("Generator ...........")
        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            # Dense layer 1
            w1 = tf.get_variable('g_w1', [self.latent_size, 128 * 16 * 16], initializer=tf.truncated_normal_initializer(
                stddev=0.02))
            b1 = tf.get_variable('g_b1', [128 * 16 * 16], initializer=tf.constant_initializer(0))

            dense1 = tf.nn.relu(tf.matmul(x, w1) + b1)
            dense1 = tf.reshape(dense1, [self.batch_size, 16, 16, 128])

            # 256 - 128 - 64 - 32 - 3
            #  64 - 32  - 16 -  8 - 4

            # Deconv layer 1
            o_shape1 = [self.batch_size, 32, 32, 128]
            w2 = tf.get_variable('g_w2', [4, 4, 128, 128], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b2 = tf.get_variable('g_b2', [128], initializer=tf.constant_initializer(0.1))
            deconv1 = self.deconv2d(dense1, w2, o_shape1) + b2
            deconv1 = tf.contrib.layers.batch_norm(inputs=deconv1, decay=0.9, is_training=True,
                                                   scope="g_bn1")

            deconv1 = tf.nn.relu(deconv1)

            # Deconv layer 2
            o_shape2 = [self.batch_size, 64, 64, 64]
            w3 = tf.get_variable('g_w3', [4, 4, 64, 128], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b3 = tf.get_variable('g_b3', [64], initializer=tf.constant_initializer(0.1))
            deconv2 = self.deconv2d(deconv1, w3, o_shape2) + b3
            deconv2 = tf.contrib.layers.batch_norm(inputs=deconv2, decay=0.9, is_training=True,
                                                   scope="g_bn2")

            deconv2 = tf.nn.relu(deconv2)

            # Deconv layer 3
            # o_shape3 = [self.batch_size, 64, 64, 3]
            w4 = tf.get_variable('g_w4', [4, 4, 64, 3], initializer=tf.truncated_normal_initializer(
                stddev=0.01))
            b4 = tf.get_variable('g_b4', [3], initializer=tf.constant_initializer(0.1))
            deconv3 = self.conv2d(deconv2, w4) + b4

            outputs = tf.nn.tanh(deconv3)

        return outputs

    def build(self):
        # real => images from database
        self.Dx = self.discriminator(self.X, reuse=False)
        self.Gz = self.generator(self.Z, reuse=False)
        # fake => images generated by gaussian distribution
        self.DG = self.discriminator(self.Gz, reuse=True)

        # generator loss => back propagate of the quality of fake images
        self.g_loss = -tf.reduce_mean(self.DG)

        # discriminator => back propagate of the combined performance of distinguishing real and fake images
        self.d_real = tf.reduce_mean(self.Dx)
        self.d_fake = tf.reduce_mean(self.DG)

        # create a distribution for gradient penalty
        penalty_dist = tf.random_uniform(shape=[self.batch_size, 1], minval=0, maxval=1)

        # Get gradient_penalty
        differences = self.Gz - self.X
        interpolates = self.X + penalty_dist * differences
        grads = tf.gradients(self.discriminator(interpolates, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

        # GRADIENT PENALTY can be viewed as a sort of regularization

        self.d_loss = self.d_fake - self.d_real + self.Lambda * gradient_penalty

        d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]

        with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
            print("reuse or not: {}".format(tf.get_variable_scope().reuse))
            assert tf.get_variable_scope().reuse == False, "Houston tengo un problem"

            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999).minimize(
                self.d_loss, var_list=d_vars)

            self.trainerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999).minimize(
                self.g_loss, var_list=g_vars)

        self.saver = tf.train.Saver(tf.global_variables())

