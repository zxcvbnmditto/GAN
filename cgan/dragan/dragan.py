import tensorflow as tf
import ops

class DRAGAN():
    def __init__(self, learning_rate, batch_size, latent_size, img_size, vocab_size, res_block_size):
        print("Constructing DRAGAN model ........")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.res_block_size = res_block_size
        self.Lambda = 5
        self.correct_imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.wrong_imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.noise = tf.placeholder(tf.float32, shape=[None, self.latent_size])
        self.correct_tags = tf.placeholder(tf.float32, shape=[None, self.vocab_size])
        self.wrong_tags = tf.placeholder(tf.float32, shape=[None, self.vocab_size])
        self.training = tf.placeholder(tf.bool)
        self.build()

    def discriminator(self, x, reuse=False):
        print("Discriminator ...........")
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            inputs = x
            outputs = ops.d_block_1(inputs, 32, 4, 2)
            for _ in range(2):
                outputs = ops.d_block_2(outputs, 32, 3, 1)

            outputs = ops.d_block_1(outputs, 64, 4, 2)
            for _ in range(4):
                outputs = ops.d_block_2(outputs, 64, 3, 1)

            outputs = ops.d_block_1(outputs, 128, 4, 2)
            for _ in range(4):
                outputs = ops.d_block_2(outputs, 128, 3, 1)

            outputs = ops.d_block_1(outputs, 256, 4, 2)
            for _ in range(4):
                outputs = ops.d_block_2(outputs, 256, 3, 1)

            # outputs = ops.d_block_1(outputs, 512, 4, 2)
            # for _ in range(4):
            #     outputs = ops.d_block_2(outputs, 512, 3, 1)

            outputs = ops.d_block_1(outputs, 512, 4, 2)
            outputs = tf.reshape(outputs, [-1, 2*2*512])

            logits = tf.layers.dense(outputs, 1, trainable=True)
            labels = tf.layers.dense(outputs, units=self.vocab_size, trainable=True)



        return logits, labels

    def generator(self, x, c, reuse=False):
        print("Generator ...........")

        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            inputs = tf.concat([x, c], axis=-1)

            # Dense layer 1
            dense1 = tf.layers.dense(inputs, units=64*8*8, trainable=True)
            dense1 = tf.contrib.layers.batch_norm(dense1, decay=0.9, epsilon=1e-5, scale=True, is_training=self.training, trainable=True)
            dense1 = tf.nn.relu(dense1)
            dense1 = tf.reshape(dense1, [-1, 8, 8, 64])

            outputs = dense1
            for i in range(self.res_block_size):
                outputs = ops.g_res_block(outputs, 64, 3, 1, self.training)

            outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, epsilon=1e-5, scale=True, is_training=self.training, trainable=True)
            outputs = tf.nn.relu(outputs)
            outputs = outputs + dense1

            for i in range(3):
                outputs = tf.layers.conv2d(outputs, filters=256, kernel_size=3, strides=1, padding='same')
                outputs = ops.pixelShuffler(outputs, 2)
                outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, epsilon=1e-5, scale=True, is_training=self.training, trainable=True)
                outputs = tf.nn.relu(outputs)

            outputs = tf.layers.conv2d(outputs, filters=3, kernel_size=9, strides=1)
            outputs = tf.nn.tanh(outputs)

        return outputs

    def build(self):
        self.fake_imgs = self.generator(self.noise, self.correct_tags, reuse=False)

        self.c_logits, self.c_labels = self.discriminator(self.correct_imgs, reuse=False)
        self.f_logits, self.f_labels = self.discriminator(self.fake_imgs, reuse=True)
        self.w_logits, self.w_labels = self.discriminator(self.wrong_imgs, reuse=True)

        # DRAGAN
        self.g_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logits, labels=tf.ones_like(self.f_logits)))

        self.d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.c_logits, labels=tf.ones_like(self.c_logits)))
        self.d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logits, labels=tf.zeros_like(self.f_logits)))
        self.d_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.w_logits, labels=tf.zeros_like(self.w_logits)))

        self.c_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.c_labels, labels=self.correct_tags)))
        self.c_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_labels, labels=self.correct_tags)))
        self.c_wrong = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.w_labels, labels=self.wrong_tags)))

        self.g_loss = self.Lambda*self.g_fake + self.c_fake
        self.d_loss = self.Lambda*self.d_real + self.c_real + self.Lambda*self.d_wrong
        # self.d_loss = self.Lambda*self.d_real + self.c_real + self.Lambda*self.d_wrong + self.c_wrong

        # self.c_loss = self.c_real + self.c_fake

        self.d_sumop = tf.summary.scalar('d_loss', self.d_loss)
        self.g_sumop = tf.summary.scalar('g_loss', self.g_loss)
        # self.c_sumop = tf.summary.scalar('c_loss', self.c_loss)

        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        # c_vars = [var for var in tf.trainable_variables() if 'classifier' in var.name or 'discriminator' in var.name or 'generator' in var.name]

        # contains the moving_mean and moving_variance, which are not in g_vars
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # update update_ops before processing the D/G trainer
        with tf.control_dependencies(update_ops):
            self.trainerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.g_loss, var_list=g_vars)

            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.d_loss, var_list=d_vars)

            # self.trainerC = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
            #     self.c_loss, var_list=c_vars)

        self.saver = tf.train.Saver(tf.global_variables())

