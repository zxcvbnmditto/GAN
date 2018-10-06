import tensorflow as tf
import math

class WGAN_GP():
    def __init__(self, FLAGS):
        print("Constructing WGAN model ........")
        # Fixed => used in discriminator
        # Moving => used in generator
        self.fixed_batch_size = FLAGS.fixed_batch_size
        self.latent_size = FLAGS.latent_size
        self.img_size = FLAGS.img_size
        self.Lambda = 10

        # Placeholders
        self.X = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, 3])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.latent_size])
        self.moving_batch_size = tf.placeholder(tf.int32)
        self.training = tf.placeholder(tf.bool)
        self.d_lr = tf.placeholder(tf.float32)
        self.g_lr = tf.placeholder(tf.float32)

        # Define variables for easy-looping over conv and conv_transpose layers
        self.kernal_size = FLAGS.kernel_size
        self.min_img_size = FLAGS.min_img_size
        self.filters = [FLAGS.img_channel]
        for i in range(int(math.log(self.img_size / self.min_img_size, 2))):
            self.filters.append(self.img_size * (2 ** i))

        self.build()

    def conv2d(self, x, w):
        return tf.nn.conv2d(input=x, filter=w, strides=[1, 2, 2, 1], padding='SAME')

    def deconv2d(self, x, w, shape):
        return tf.nn.conv2d_transpose(value=x, filter=w, output_shape=shape, strides=[1, 2, 2, 1], padding='SAME')

    def block_D(self, i, inputs):
        in_size = self.filters[i]
        out_size = self.filters[i+1]

        print("D: ", i, in_size, out_size)

        w = tf.get_variable('d_w_conv{:d}'.format(i),
                            [self.kernal_size, self.kernal_size, in_size, out_size],
                            initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable('d_b_conv{:d}'.format(i),
                            [out_size],
                            initializer=tf.constant_initializer(0))

        outputs = tf.add(self.conv2d(inputs, w), b, name='conv_{:d}'.format(i))
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2, name='LR_{:d}'.format(i))

        return outputs

    def block_G(self, i, inputs, batch_norm=True, activation='relu'):
        in_size = self.filters[-(i+1)]
        out_size = self.filters[-(i+2)]
        img_size = self.min_img_size * (2 ** (i+1))
        output_shape = [self.moving_batch_size, img_size, img_size, out_size]

        print("G: ", i, in_size, out_size, img_size)

        w = tf.get_variable('g_w_conv{:d}'.format(i),
                            [self.kernal_size, self.kernal_size, out_size, in_size],
                            initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable('g_b_conv{:d}'.format(i),
                            [out_size],
                            initializer=tf.constant_initializer(0))
        outputs = tf.add(self.deconv2d(inputs, w, output_shape), b, name="deconv_{:d}".format(i))

        if batch_norm:
            outputs = tf.contrib.layers.batch_norm(outputs, scope='g_bn_conv{:d}'.format(i), decay=0.9, epsilon=1e-5, scale=True,
                                               is_training=self.training)

        if activation == 'relu':
            outputs = tf.nn.relu(outputs, name="R_{:d}".format(i))
        elif activation == 'tanh':
            outputs = tf.nn.tanh(outputs, name="outputs")

        return outputs

    def discriminator(self, x, reuse=False):
        print("Discriminator ...........")
        with tf.variable_scope('Discriminator', reuse=reuse):
            conv_outputs = x

            # Conv Layers
            for i in range(len(self.filters) - 1):
                conv_outputs = self.block_D(i, conv_outputs)

            # Flatten / Reshape
            flatten = tf.reshape(conv_outputs, [self.fixed_batch_size, self.min_img_size * self.min_img_size * self.filters[-1]])

            # Dense output layer
            d_w = tf.get_variable('d_w_dense', [self.min_img_size * self.min_img_size * self.filters[-1], 1], initializer=tf.random_normal_initializer(stddev=0.02))
            d_b = tf.get_variable('d_b_dense', [1], initializer=tf.constant_initializer(0))
            outputs = tf.add(tf.matmul(flatten, d_w), d_b, name="outputs")

        return outputs

    def generator(self, x, reuse=False):
        print("Generator ...........")
        with tf.variable_scope('Generator', reuse=reuse) as scope:
            # Dense input layer
            g_w_dense = tf.get_variable('g_w_dense', [self.latent_size, self.filters[-1] * self.min_img_size * self.min_img_size],
                                        initializer=tf.random_normal_initializer(stddev=0.02))
            g_b_dense = tf.get_variable('g_b_dense', [self.filters[-1] * self.min_img_size * self.min_img_size],
                                        initializer=tf.constant_initializer(0))
            dense = tf.add(tf.matmul(x, g_w_dense), g_b_dense, name="dense")
            dense = tf.contrib.layers.batch_norm(dense, scope='g_bn_dense', decay=0.9, epsilon=1e-5, scale=True,
                                                  is_training=self.training)
            dense = tf.nn.relu(dense, "R_dense")

            # Reshape Dense to Conv
            outputs = tf.reshape(dense, [-1, self.min_img_size, self.min_img_size, self.filters[-1]])

            # Conv Layers
            for i in range(len(self.filters) - 1):
                # Disable batch_norm and use tanh for the last block
                if i == len(self.filters) - 2:
                    outputs = self.block_G(i, outputs, False, 'tanh')
                else:
                    outputs = self.block_G(i, outputs)

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
        penalty_dist = tf.random_uniform(shape=[self.img_size, 1], minval=0, maxval=1)

        # Get gradient_penalty
        differences = self.Gz - self.X
        interpolates = self.X + penalty_dist * differences
        grads = tf.gradients(self.discriminator(interpolates, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

        # GRADIENT PENALTY can be viewed as a sort of regularization
        self.d_loss = self.d_fake - self.d_real + self.Lambda * self.gradient_penalty

        with tf.variable_scope("Summary"):
            self.d_sumop = tf.summary.scalar('d_loss', self.d_loss)
            self.g_sumop = tf.summary.scalar('g_loss', self.g_loss)
            self.p_sumop = tf.summary.scalar('penalty', self.gradient_penalty)

        d_vars = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            with tf.variable_scope("Generator_Trainer"):
                self.trainerG = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=0.5, beta2=0.9).minimize(
                    self.g_loss, var_list=g_vars)

        with tf.variable_scope("Discriminator_Trainer"):
            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=0.5, beta2=0.9).minimize(
                self.d_loss, var_list=d_vars)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)


