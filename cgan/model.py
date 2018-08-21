import tensorflow as tf

class CGAN():
    def __init__(self, learning_rate, batch_size, latent_size, img_size, vocab_size):
        print("Constructing CGAN model ........")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.Lambda = 10
        self.correct_imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.wrong_imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.noise = tf.placeholder(tf.float32, shape=[None, self.latent_size])
        self.correct_tags = tf.placeholder(tf.float32, shape=[None, 2*self.vocab_size])
        self.wrong_tags = tf.placeholder(tf.float32, shape=[None, 2*self.vocab_size])
        self.training = tf.placeholder(tf.bool)
        self.build()

    def discriminator(self, x, c, reuse=False):
        print("Discriminator ...........")
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            tags_embed = tf.layers.dense(c, units=10, trainable=True)
            tags_embed = tf.reshape(tags_embed, [-1, 1, 1, 10])
            tags_embed = tf.tile(tags_embed, [1, 8, 8, 1])

            # 64 * 64 * 3
            # Conv layer 1
            conv_1 = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2, padding='same')
            conv_1 = tf.contrib.layers.batch_norm(conv_1, scope='bn1', decay=0.9, epsilon=1e-5, scale=True,
                                                  is_training=self.training, trainable=True)
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
            # Conv layer 5
            conv_5 = tf.layers.conv2d(tf.concat([conv_3, tags_embed], axis=-1), filters=128, kernel_size=5, strides=1, padding='same')
            conv_5 = tf.nn.leaky_relu(conv_5, alpha=0.2)

            # 8*8*128
            # flatten
            flatten = tf.layers.flatten(conv_5)

            # Fully connect layer
            dense1 = tf.layers.dense(flatten, units=1, trainable=True)

            outputs = dense1

        return outputs

    def generator(self, x, c, reuse=False):
        print("Generator ...........")

        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            w1 = tf.get_variable('w1', shape=[self.vocab_size * 2, 10], initializer=tf.random_normal_initializer(stddev=0.02))
            b1 = tf.get_variable('b1', shape=[10], initializer=tf.constant_initializer(0))
            tags_embed = tf.matmul(c, w1) + b1
            tags_embed = tf.contrib.layers.batch_norm(tags_embed, scope='bn1', decay=0.9, epsilon=1e-5, scale=True,
                                                  is_training=self.training, trainable=True)

            inputs = tf.concat([x, tags_embed], axis=-1)

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

        # Four cases
        # Case 1 => correct_imgs + correct_tags
        # Case 2 => wrong_imgs (randomly sampled from db) + correct_tags
        # Case 3 => fake_imgs (generated by the generator) + correct_tags
        # Case 4 => correct_imgs + wrong_tags (randomly sampled from db)
        self.case1 = self.discriminator(self.correct_imgs, self.correct_tags, reuse=False)
        self.case2 = self.discriminator(self.wrong_imgs, self.correct_tags, reuse=True)
        self.case3 = self.discriminator(self.fake_imgs, self.correct_tags, reuse=True)
        self.case4 = self.discriminator(self.correct_imgs, self.wrong_tags, reuse=True)

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.case3, labels=tf.ones_like(self.case3)))

        self.d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.case1, labels=tf.ones_like(self.case1)) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.case2, labels=tf.zeros_like(self.case2)) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.case3, labels=tf.zeros_like(self.case3)) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.case4, labels=tf.zeros_like(self.case4))
        )

        self.d_sumop = tf.summary.scalar('d_loss', self.d_loss)
        self.g_sumop = tf.summary.scalar('g_loss', self.g_loss)

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

        self.saver = tf.train.Saver(tf.global_variables())

