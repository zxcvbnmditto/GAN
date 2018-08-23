import tensorflow as tf
import ops

class ACGAN():
    def __init__(self, learning_rate, batch_size, latent_size, img_size, vocab_size):
        print("Constructing ACGAN model ........")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.Lambda = 10
        self.correct_imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.noise = tf.placeholder(tf.float32, shape=[None, self.latent_size])
        self.correct_tags = tf.placeholder(tf.float32, shape=[None, self.vocab_size])
        self.training = tf.placeholder(tf.bool)
        self.build()

    def discriminator(self, x, reuse=False):
        print("Discriminator ...........")
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            # 64*64*3
            # Conv layer 1
            conv_1 = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same')
            conv_1 = tf.nn.leaky_relu(conv_1, alpha=0.2)

            # 32*32*32
            # Conv layer 2
            conv_2 = tf.layers.conv2d(conv_1, filters=64, kernel_size=5, strides=2, padding='same')
            conv_2 = tf.contrib.layers.batch_norm(conv_2, scope='bn2', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            conv_2 = tf.nn.leaky_relu(conv_2, alpha=0.2)

            # 16*16*64
            # Conv layer 3
            conv_3 = tf.layers.conv2d(conv_2, filters=128, kernel_size=5, strides=2, padding='same')
            conv_3 = tf.contrib.layers.batch_norm(conv_3, scope='bn3', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            conv_3 = tf.nn.leaky_relu(conv_3, alpha=0.2)

            # 8*8*128
            # Conv layer 4
            conv_4 = tf.layers.conv2d(conv_3, filters=256, kernel_size=5, strides=2, padding='same')
            conv_4 = tf.contrib.layers.batch_norm(conv_4, scope='bn4', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            conv_4 = tf.nn.leaky_relu(conv_4, alpha=0.2)

            # 4*4*256
            # Conv layer 5
            conv_5 = tf.layers.conv2d(conv_4, filters=256, kernel_size=5, strides=1, padding='same')
            conv_5 = tf.nn.leaky_relu(conv_5, alpha=0.2)

            # eval if an image is real or fake
            flatten = tf.layers.flatten(conv_5)
            real_fake_logit = tf.layers.dense(flatten, units=1, trainable=True)

        return real_fake_logit, conv_4

    def classifier(self, x, reuse=False):
        with tf.variable_scope('classifier') as scope:
            if (reuse):
                scope.reuse_variables()
            # x => input => a conv layer
            flatten = tf.layers.flatten(x)
            # Dense 1
            dense1 = tf.layers.dense(flatten, units=self.vocab_size, trainable=True)

        return dense1


    def generator(self, x, c, reuse=False):
        print("Generator ...........")

        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            inputs = tf.concat([x, c], axis=-1)

            # Dense layer 1
            dense1 = tf.layers.dense(inputs, units=256*4*4, trainable=True)
            dense1 = tf.contrib.layers.batch_norm(dense1, scope='bn2', decay=0.9, epsilon=1e-5, scale=True,
                                                  is_training=self.training, trainable=True)
            dense1 = tf.nn.relu(dense1)
            dense1 = tf.reshape(dense1, [-1, 4, 4, 256])

            # Deconv layer 1
            deconv1 = tf.layers.conv2d_transpose(dense1, filters=128, kernel_size=5, strides=2, padding='same')
            deconv1 = tf.contrib.layers.batch_norm(deconv1, scope='bn3', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            deconv1 = tf.nn.relu(deconv1)

            # Deconv layer 2
            deconv2 = tf.layers.conv2d_transpose(deconv1, filters=64, kernel_size=5, strides=2, padding='same')
            deconv2 = tf.contrib.layers.batch_norm(deconv2, scope='bn4', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            deconv2 = tf.nn.relu(deconv2)

            # Deconv layer 3
            deconv3 = tf.layers.conv2d_transpose(deconv2, filters=32, kernel_size=5, strides=2, padding='same')
            deconv3 = tf.contrib.layers.batch_norm(deconv3, scope='bn5', decay=0.9, epsilon=1e-5, scale=True,
                                                   is_training=self.training, trainable=True)
            deconv3 = tf.nn.relu(deconv3)

            # Deconv layer 4
            deconv4 = tf.layers.conv2d_transpose(deconv3, filters=3, kernel_size=5, strides=2, padding='same')

            outputs = tf.nn.tanh(deconv4)

        return outputs

    def build(self):
        self.fake_imgs = self.generator(self.noise, self.correct_tags, reuse=False)

        self.c_real_label , self.dc_net = self.discriminator(self.correct_imgs, reuse=False)
        self.f_real_label, self.df_net = self.discriminator(self.fake_imgs, reuse=True)
        self.c_tag_class = self.classifier(self.dc_net, reuse=False)
        self.f_tag_class = self.classifier(self.df_net, reuse=True)

        # ACGAN
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_real_label, labels=tf.ones_like(self.f_real_label)))

        self.d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.c_real_label, labels=tf.ones_like(self.c_real_label)))
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_real_label, labels=tf.zeros_like(self.f_real_label)))
        self.d_loss = self.d_real + self.d_loss

        self.c_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.c_tag_class, labels=self.correct_tags))
        self.c_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.f_tag_class, labels=self.correct_tags))
        self.c_loss = self.c_real + self.c_fake

        self.d_sumop = tf.summary.scalar('d_loss', self.d_loss)
        self.g_sumop = tf.summary.scalar('g_loss', self.g_loss)
        self.c_sumop = tf.summary.scalar('c_loss', self.c_loss)

        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        c_vars = [var for var in tf.trainable_variables() if 'classifier' in var.name or 'discriminator' in var.name or 'generator' in var.name]

        # contains the moving_mean and moving_variance, which are not in g_vars
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # update update_ops before processing the D/G trainer
        with tf.control_dependencies(update_ops):
            self.trainerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.g_loss, var_list=g_vars)

            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.d_loss, var_list=d_vars)

            self.trainerC = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.c_loss, var_list=c_vars)

        self.saver = tf.train.Saver(tf.global_variables())

