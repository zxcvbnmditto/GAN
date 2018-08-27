import tensorflow as tf
import ops

def resblock1_G(inputs, filters, kernel_size, stride, training, norm, scope):
    with tf.variable_scope(scope):
        outputs = ops.conv_2d(inputs, filters, kernel_size, stride, padding='SAME', stddev=0.02, norm=norm, training=training, scope="conv_1")
        outputs = tf.nn.relu(outputs)
        outputs = ops.conv_2d(outputs, filters, kernel_size, stride, padding='SAME', stddev=0.02, norm=norm, training=training, scope="conv_2")
        outputs = outputs + inputs

    return outputs

def resblock2_G(inputs, filters, kernel_size, stride, scale, training, norm, scope):
    with tf.variable_scope(scope):
        outputs = ops.conv_2d(inputs, filters, kernel_size, stride, padding='SAME', stddev=0.02, norm=norm, training=training, scope="conv")
        outputs = ops.pixelShuffler(outputs, scale)
        outputs = tf.nn.relu(outputs)

    return outputs

# for sizing up
def resblock_D(inputs, filters, kernel_size, stride, training, norm, scope):
    with tf.variable_scope(scope):
        outputs = ops.conv_2d(inputs, filters, kernel_size, stride, padding='SAME', stddev=0.02, norm=norm, training=training, scope='conv')
        outputs = tf.nn.leaky_relu(outputs, alpha=0.01)

    return outputs


class STARGAN():
    def __init__(self, learning_rate, batch_size, latent_size, img_size, vocab_size, res_block_size):
        print("Constructing STARGAN model ........")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.res_block_size = res_block_size
        self.d_norm = None
        self.g_norm = "instance_norm"
        self.lambda_cls = 1
        self.lambda_recon = 10
        self.lambda_gp = 10

        self.real_imgs = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
        self.origin_tags = tf.placeholder(tf.float32, shape=[None, self.vocab_size])
        self.target_tags = tf.placeholder(tf.float32, shape=[None, self.vocab_size])

        self.training = tf.placeholder(tf.bool)
        self.build()

    def discriminator(self, x, reuse=False):
        print("Discriminator ...........")
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            inputs = x

            outputs = resblock_D(inputs, 16, 4, 2, self.training, self.d_norm, scope='d_0')
            outputs = resblock_D(outputs, 32, 4, 2, self.training, self.d_norm, scope='d_1')
            outputs = resblock_D(outputs, 64, 4, 2, self.training, self.d_norm, scope='d_2')
            outputs = resblock_D(outputs, 128, 4, 2, self.training, self.d_norm, scope='d_3')
            outputs = resblock_D(outputs, 256, 4, 2, self.training, self.d_norm, scope='d_4')
            outputs = resblock_D(outputs, 512, 4, 2, self.training, self.d_norm, scope='d_5')
            outputs = resblock_D(outputs, 1024, 4, 2, self.training, self.d_norm, scope='d_6')

            # outputs = tf.reshape(outputs, [-1, 1 * 1 * 1024])
            # logits = ops.dense(outputs, 1, 0.02, self.training, None, scope="d_logits")
            # labels = ops.dense(outputs, self.vocab_size, 0.02, self.training, None, scope="d_labels")

            logits = ops.conv_2d(outputs, 1, 3, 1, padding='SAME', stddev=0.02, training=self.training, norm=self.d_norm, scope="d_logits")
            logits = tf.reshape(logits, [-1, 1])
            print(logits)

            labels = ops.conv_2d(outputs, self.vocab_size, 7, 1, padding='SAME', stddev=0.02, training=self.training, norm=self.d_norm, scope="d_labels")
            labels = tf.reshape(labels, [-1, self.vocab_size])
            print(labels)

        return logits, labels

    def generator(self, x, c, reuse=False):
        print("Generator ...........")

        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            c = tf.reshape(c, [-1, 1, 1, self.vocab_size])
            c = tf.tile(c, [1, 128, 128, 1])
            inputs = tf.concat([x, c], axis=3)
            # print(inputs.shape)

            # Downsizing
            conv1 = ops.conv_2d(inputs, 16, 3, 2, padding='SAME', stddev=0.02, training=self.training, norm=self.g_norm, scope="g_downsize_1")
            conv1 = tf.nn.relu(conv1)

            conv1 = ops.conv_2d(conv1, 32, 3, 2, padding='SAME', stddev=0.02, training=self.training, norm=self.g_norm, scope="g_downsize_2")
            conv1 = tf.nn.relu(conv1)

            # inputs = tf.concat([conv1, c], axis=3)

            conv1 = ops.conv_2d(conv1, 64, 3, 2, padding='SAME', stddev=0.02, training=self.training, norm=self.g_norm, scope="g_downsize_3")
            conv1 = tf.nn.relu(conv1)
     
            outputs = conv1
            # ResBlock
            for i in range(self.res_block_size):
                outputs = resblock1_G(outputs, 64, 3, 1, self.training, norm=self.g_norm, scope='g_residual_{:d}'.format(i))

            # Upsizing by pixel shuffling
            for i in range(3):
                outputs = resblock2_G(outputs, 256, 3, 1, 2, self.training, norm=self.g_norm, scope='g_upsize_{:d}'.format(i))

            outputs = ops.conv_2d(outputs, 3, 9, 1, padding='SAME', stddev=0.02, training=self.training, norm=None, scope="g_conv_last")
            outputs = tf.nn.tanh(outputs)

        return outputs

    def build(self):
        self.fake_imgs = self.generator(self.real_imgs, self.target_tags, reuse=False)
        self.recon_imgs = self.generator(self.fake_imgs, self.origin_tags, reuse=True)


        self.real_logits, self.real_labels = self.discriminator(self.real_imgs, reuse=False)
        self.fake_logits, self.fake_labels = self.discriminator(self.fake_imgs, reuse=True)

        # STARGAN

        # Logit Loss
        self.l_real = tf.reduce_mean(self.real_logits)
        self.l_fake = tf.reduce_mean(self.fake_logits)
        self.l_recon = tf.reduce_mean(tf.abs(self.real_imgs - self.recon_imgs))

        # Label Loss
        self.c_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_labels, labels=self.origin_tags), axis=1))
        self.c_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_labels, labels=self.target_tags), axis=1))

        # Penalty Loss
        penalty_dist = tf.random_uniform(shape=[self.batch_size, 128, 128, 3], minval=0, maxval=1)
        differences = self.fake_imgs - self.real_imgs
        interpolates = self.real_imgs + penalty_dist * differences
        interpolates_logits, _ = self.discriminator(interpolates, reuse=True)
        grads = tf.gradients(interpolates_logits, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

        self.g_loss = -self.l_fake + self.lambda_recon*self.l_recon + self.lambda_cls*self.c_fake
        self.d_loss = self.l_fake - self.l_real + self.lambda_cls*self.c_real + self.lambda_gp*self.gradient_penalty

        self.d_sumop = tf.summary.scalar('d_loss', self.d_loss)
        self.g_sumop = tf.summary.scalar('g_loss', self.g_loss)
        # self.lr_sumop = tf.summary.scalar('l_real', self.l_real)
        # self.cr_sumop = tf.summary.scalar('c_real', self.c_real)
        # self.lf_sumop = tf.summary.scalar('l_fake', self.l_fake)
        # self.cf_sumop = tf.summary.scalar('c_fake', self.c_fake)
        # self.lrecon_sumop = tf.summary.scalar('l_recon', self.l_recon)


        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # update update_ops before processing the D/G trainer
        with tf.control_dependencies(update_ops):
            self.trainerG = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.g_loss, var_list=g_vars)

            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9).minimize(
                self.d_loss, var_list=d_vars)

        self.saver = tf.train.Saver(tf.global_variables())

