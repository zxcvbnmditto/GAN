import tensorflow as tf

class WGAN_GP():
    def __init__(self, learning_rate, batch_size, latent_size, img_size):
        print("Constructing WGAN model ........")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.img_size = img_size
        self.Lambda = 10
        self.X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.latent_size])
        self.training = tf.placeholder(tf.bool)
        self.build()

    def discriminator(self, x, reuse=False):
        print("Discriminator ...........")
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            # Conv layer 1
            conv_1 = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='same')
            # conv_1 = tf.contrib.layers.layer_norm(conv_1, trainable=True, scope='ln1')
            conv_1 = tf.nn.leaky_relu(conv_1, alpha=0.2)

            # Conv layer 2
            conv_2 = tf.layers.conv2d(conv_1, filters=128, kernel_size=5, strides=2, padding='same')
            conv_2 = tf.contrib.layers.layer_norm(conv_2, trainable=True, scope='ln2')
            conv_2 = tf.nn.leaky_relu(conv_2, alpha=0.2)

            # Conv layer 3
            conv_3 = tf.layers.conv2d(conv_2, filters=256, kernel_size=5, strides=2, padding='same')
            conv_3 = tf.contrib.layers.layer_norm(conv_3, trainable=True, scope='ln3')
            conv_3 = tf.nn.leaky_relu(conv_3, alpha=0.2)

            # Conv layer 4
            conv_4 = tf.layers.conv2d(conv_3, filters=512, kernel_size=5, strides=2, padding='same')
            conv_4 = tf.contrib.layers.layer_norm(conv_4, trainable=True, scope='ln4')
            conv_4 = tf.nn.leaky_relu(conv_4, alpha=0.2)

            # flatten
            flatten = tf.layers.flatten(conv_3)

            # Fully connect layer
            dense1 = tf.layers.dense(flatten, units=1, trainable=True)

            # Float
            outputs = dense1

        return outputs

    def generator(self, x, reuse=False):
        print("Generator ...........")

        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            # Dense layer 1
            dense1 = tf.layers.dense(x, units=512*4*4, trainable=True)
            dense1 = tf.contrib.layers.batch_norm(dense1, scope='bn1', decay=0.9, epsilon=1e-5, scale=True,
                                                  is_training=self.training, trainable=True)
            dense1 = tf.nn.relu(dense1)
            dense1 = tf.reshape(dense1, [-1, 4, 4, 512])

            # Deconv layer 1
            deconv1 = tf.layers.conv2d_transpose(dense1, filters=256, kernel_size=5, strides=2, padding='same')
            deconv1 = tf.contrib.layers.batch_norm(deconv1, scope='bn2', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            deconv1 = tf.nn.relu(deconv1)

            # Deconv layer 2
            deconv2 = tf.layers.conv2d_transpose(deconv1, filters=128, kernel_size=5, strides=2, padding='same')
            deconv2 = tf.contrib.layers.batch_norm(deconv2, scope='bn3', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            deconv2 = tf.nn.relu(deconv2)

            # Deconv layer 3
            deconv3 = tf.layers.conv2d_transpose(deconv2, filters=64, kernel_size=5, strides=2, padding='same')
            deconv3 = tf.contrib.layers.batch_norm(deconv3, scope='bn4', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            deconv3 = tf.nn.relu(deconv3)

            # Deconv layer 4
            deconv4 = tf.layers.conv2d_transpose(deconv3, filters=3, kernel_size=5, strides=2, padding='same')

            outputs = tf.nn.tanh(deconv4)

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
        self.gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

        # GRADIENT PENALTY can be viewed as a sort of regularization

        self.d_loss = self.d_fake - self.d_real + self.Lambda * self.gradient_penalty

        self.d_sumop = tf.summary.scalar('d_loss', self.d_loss)
        self.g_sumop = tf.summary.scalar('g_loss', self.g_loss)
        self.p_sumop = tf.summary.scalar('penalty', self.gradient_penalty)

        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        # contains the moving_mean and moving_variance, which are not in g_vars
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # update update_ops before processing the D/G trainer
        with tf.control_dependencies(update_ops):
            self.trainerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.g_loss, var_list=g_vars)

            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.d_loss, var_list=d_vars)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
