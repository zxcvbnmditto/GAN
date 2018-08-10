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
        self.training = tf.placeholder(tf.bool)
        self.build()

    # filter shape = [filter_width, filter_height, channel (prev filter number), output_filter_num]
    # stride shape = [1, width_stride, height_stride, 1]
    def conv2d(self, x, w):
        return tf.nn.conv2d(input=x, filter=w, strides=[1, 2, 2, 1], padding='SAME')

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
            w1 = tf.get_variable('d_w1', [5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('d_b1', [64], initializer=tf.constant_initializer(0))
            conv_1 = self.conv2d(x, w1) + b1
            conv_1 = tf.nn.leaky_relu(conv_1, alpha=0.2)
            conv_1 = tf.contrib.layers.layer_norm(conv_1, scope='d_ln1')

            # Conv layer 2
            w2 = tf.get_variable('d_w2', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable('d_b2', [128], initializer=tf.constant_initializer(0))
            conv_2 = self.conv2d(conv_1, w2) + b2
            conv_2 = tf.nn.leaky_relu(conv_2, alpha=0.2)
            conv_2 = tf.contrib.layers.layer_norm(conv_2, scope='d_ln2')

            # Conv layer 3
            w3 = tf.get_variable('d_w3', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b3 = tf.get_variable('d_b3', [256], initializer=tf.constant_initializer(0))
            conv_3 = self.conv2d(conv_2, w3) + b3
            conv_3 = tf.nn.leaky_relu(conv_3, alpha=0.2)
            conv_3 = tf.contrib.layers.layer_norm(conv_3, scope='d_ln3')

            # Conv layer 4
            w4 = tf.get_variable('d_w4', [5, 5, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b4 = tf.get_variable('d_b4', [512], initializer=tf.constant_initializer(0))
            conv_4 = self.conv2d(conv_3, w4) + b4
            conv_4 = tf.nn.leaky_relu(conv_4, alpha=0.2)
            conv_4 = tf.contrib.layers.layer_norm(conv_4, scope='d_ln4')

            flatten = tf.reshape(conv_4, [self.batch_size, 4 * 4 * 512])

            # Dense layer 1
            w5 = tf.get_variable('d_w5', [4 * 4 * 512, 1], initializer=tf.random_normal_initializer(stddev=0.02))
            b5 = tf.get_variable('d_b5', [1], initializer=tf.constant_initializer(0))
            dense1 = tf.matmul(flatten, w5) + b5

            outputs = dense1

        return outputs

    def generator(self, x, reuse=False, training=True):
        print("Generator ...........")

        def reset_batch(size):
            is_train = True

            if size == 256:
                is_train = False

            return size, is_train

        self.batch_size, training = tf.cond(tf.equal(self.training, tf.constant(False)), lambda: reset_batch(256),
                                  lambda: reset_batch(64))

        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            # Dense layer 1
            w1 = tf.get_variable('g_w1', [self.latent_size, 512 * 4 * 4], initializer=tf.random_normal_initializer(
                stddev=0.02))
            b1 = tf.get_variable('g_b1', [512 * 4 * 4], initializer=tf.constant_initializer(0))
            dense1 = tf.nn.leaky_relu(tf.matmul(x, w1) + b1)
            dense1 = tf.reshape(dense1, [self.batch_size, 4, 4, 512])
            dense1 = tf.contrib.layers.batch_norm(dense1, scope='g_bn1', decay=0.9, epsilon=1e-5, scale=True,
                                                  is_training=training)

            # Deconv layer 1
            o_shape1 = [self.batch_size, 8, 8, 256]
            w2 = tf.get_variable('g_w2', [5, 5, 256, 512], initializer=tf.random_normal_initializer(
                stddev=0.02))
            b2 = tf.get_variable('g_b2', [256], initializer=tf.constant_initializer(0.1))
            deconv1 = self.deconv2d(dense1, w2, o_shape1) + b2
            deconv1 = tf.nn.leaky_relu(deconv1)
            deconv1 = tf.contrib.layers.batch_norm(deconv1, scope='g_bn2', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=training)

            # Deconv layer 2
            o_shape2 = [self.batch_size, 16, 16, 128]
            w3 = tf.get_variable('g_w3', [5, 5, 128, 256], initializer=tf.random_normal_initializer(
                stddev=0.02))
            b3 = tf.get_variable('g_b3', [128], initializer=tf.constant_initializer(0.1))
            deconv2 = self.deconv2d(deconv1, w3, o_shape2) + b3
            deconv2 = tf.nn.leaky_relu(deconv2)
            deconv2 = tf.contrib.layers.batch_norm(deconv2, scope='g_bn3', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=training)

            # Deconv layer 3
            o_shape3 = [self.batch_size, 32, 32, 64]
            w4 = tf.get_variable('g_w4', [5, 5, 64, 128], initializer=tf.random_normal_initializer(
                stddev=0.02))
            b4 = tf.get_variable('g_b4', [64], initializer=tf.constant_initializer(0.1))
            deconv3 = self.deconv2d(deconv2, w4, o_shape3) + b4
            deconv3 = tf.nn.leaky_relu(deconv3)
            deconv3 = tf.contrib.layers.batch_norm(deconv3, scope='g_bn4', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=training)

            # Deconv layer 4
            o_shape4 = [self.batch_size, 64, 64, 3]
            w5 = tf.get_variable('g_w5', [5, 5, 3, 64], initializer=tf.random_normal_initializer(
                stddev=0.02))
            b5 = tf.get_variable('g_b5', [3], initializer=tf.constant_initializer(0.1))
            deconv4 = self.deconv2d(deconv3, w5, o_shape4) + b5

            outputs = tf.nn.tanh(deconv4)

        self.batch_size, _ = reset_batch(64)

        return outputs

    def build(self):
        # real => images from database
        self.Dx = self.discriminator(self.X, reuse=False)
        self.Gz = self.generator(self.Z, reuse=False, training=self.training)
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
        self.gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

        # GRADIENT PENALTY can be viewed as a sort of regularization

        self.d_loss = self.d_fake - self.d_real + self.Lambda * self.gradient_penalty

        self.d_sumop = tf.summary.scalar('d_loss', self.d_loss)
        self.g_sumop = tf.summary.scalar('g_loss', self.g_loss)
        self.p_sumop = tf.summary.scalar('penalty', self.gradient_penalty)

        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

        print(d_vars)
        print(g_vars)

        with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
            print("reuse or not: {}".format(tf.get_variable_scope().reuse))
            assert tf.get_variable_scope().reuse == False, "Houston tengo un problem"

            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999).minimize(
                self.d_loss, var_list=d_vars)

            self.trainerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999).minimize(
                self.g_loss, var_list=g_vars)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

