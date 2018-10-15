import tensorflow as tf
import ops


def residule_block(inputs, filters, kernel_size, stride, scope):
    with tf.variable_scope(scope):
        outputs = ops.conv2d(inputs, filters, kernel_size, stride, scope='conv_1')
        outputs = ops.instance_norm(outputs, 'in_1')
        outputs = ops.relu(outputs)

        outputs = ops.conv2d(outputs, filters, kernel_size, stride, scope='conv_2')
        outputs = ops.instance_norm(outputs, 'in_2')

    return outputs + inputs


class STARGAN():
    def __init__(self, FLAGS):
        print("Constructing STARGAN model ........")
        self.batch_size = FLAGS.fixed_batch_size
        self.feature_length = FLAGS.feature_length
        self.res_block_size = FLAGS.res_block_size
        self.lambda_cls = FLAGS.lambda_cls
        self.lambda_recon = FLAGS.lambda_recon
        self.lambda_gp = FLAGS.lambda_gp

        self.real_imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.origin_tags = tf.placeholder(tf.float32, shape=[None, self.feature_length])
        self.target_tags = tf.placeholder(tf.float32, shape=[None, self.feature_length])
        self.d_lr = tf.placeholder(tf.float32)
        self.g_lr = tf.placeholder(tf.float32)

        self.training = tf.placeholder(tf.bool)
        self.build()

    def discriminator(self, x, reuse=False):
        print("Discriminator ...........")
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            outputs = x

            outputs = ops.lrelu(ops.conv2d(outputs, 64, 4, 2, scope="d_1"))
            outputs = ops.lrelu(ops.conv2d(outputs, 128, 4, 2, scope="d_2"))
            outputs = ops.lrelu(ops.conv2d(outputs, 256, 4, 2, scope="d_3"))
            outputs = ops.lrelu(ops.conv2d(outputs, 512, 4, 2, scope="d_4"))
            outputs = ops.lrelu(ops.conv2d(outputs, 1024, 4, 2, scope="d_5"))
            # outputs = ops.lrelu(ops.conv2d(outputs, 2048, 4, 2, scope="d_6"))

            # Logit
            logits = ops.conv2d(outputs, 1, 3, 1, scope="d_logits")

            # Label
            # pad 0 => https://arxiv.org/pdf/1711.09020.pdf
            labels = ops.conv2d(outputs, self.feature_length, 2, 1, 'VALID', scope="d_lables")
            labels = tf.reshape(labels, [-1, self.feature_length])

        return logits, labels

    def generator(self, x, c, reuse=False):
        print("Generator ...........")

        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            c = tf.reshape(c, [-1, 1, 1, self.feature_length])
            c = tf.tile(c, [1, 64, 64, 1])
            inputs = tf.concat([x, c], axis=3)
            # print(inputs)

            # Iinitial concat conv layer
            # pad 3 => https://arxiv.org/pdf/1711.09020.pdf
            outputs = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]])
            outputs = ops.conv2d(outputs, 64, 7, 1, 'VALID', scope="g_init")

            # Downsize twice
            outputs = ops.conv2d(outputs, 128, 4, 2, scope="g_down_sample_1")
            outputs = ops.conv2d(outputs, 256, 4, 2, scope="g_down_sample_2")

            # Resblock  CONV-(N256, K3x3, S1, P1), IN, ReLU
            outputs = residule_block(outputs, 256, 3, 1, scope="g_resblock_1")
            outputs = residule_block(outputs, 256, 3, 1, scope="g_resblock_2")
            outputs = residule_block(outputs, 256, 3, 1, scope="g_resblock_3")
            outputs = residule_block(outputs, 256, 3, 1, scope="g_resblock_4")
            outputs = residule_block(outputs, 256, 3, 1, scope="g_resblock_5")
            outputs = residule_block(outputs, 256, 3, 1, scope="g_resblock_6")

            # Upsampling twice
            outputs = ops.instance_norm(ops.relu(ops.deconv2d(outputs, 128, 4, 2, scope='g_upsampling_1')), 'g_in_1')
            outputs = ops.instance_norm(ops.relu(ops.deconv2d(outputs, 64, 4, 2, scope='g_upsampling_2')), 'g_in_2')

            # pad 3 => https://arxiv.org/pdf/1711.09020.pdf
            outputs = tf.pad(outputs, [[0, 0], [3, 3], [3, 3], [0, 0]])
            outputs = ops.tanh(ops.conv2d(outputs, 3, 7, 1, 'VALID', scope='g_out'))

        return outputs

    def build(self):
        self.fake_imgs = self.generator(self.real_imgs, self.target_tags, reuse=False)
        self.recon_imgs = self.generator(self.fake_imgs, self.origin_tags, reuse=True)

        self.real_logits, self.real_labels = self.discriminator(self.real_imgs, reuse=False)
        self.fake_logits, self.fake_labels = self.discriminator(self.fake_imgs, reuse=True)

        # STARGAN

        # Logit Loss
        # WGAN loss has undesired behavior, so instead apply GAN loss
        self.l_real = tf.reduce_mean(self.real_logits)
        self.l_fake = tf.reduce_mean(self.fake_logits)
        self.l_recon = tf.reduce_mean(tf.abs(self.real_imgs - self.recon_imgs))
        # self.l_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits, labels=tf.ones_like(self.real_logits)))
        # self.l_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.zeros_like(self.fake_logits)))
        # self.g_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.ones_like(self.fake_logits)))

        # Label Loss
        self.c_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_labels, labels=self.origin_tags), axis=1))
        self.c_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_labels, labels=self.target_tags), axis=1))

        # Penalty Loss
        penalty_dist = tf.random_uniform(shape=[self.batch_size*self.feature_length, 1, 1, 1], minval=0, maxval=1)
        differences = self.fake_imgs - self.real_imgs
        interpolates = self.real_imgs + penalty_dist * differences
        interpolates_logits, _ = self.discriminator(interpolates, reuse=True)
        grads = tf.gradients(interpolates_logits, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

        # self.g_loss = self.g_fake + self.lambda_recon*self.l_recon + self.lambda_cls*self.c_fake
        # self.d_loss = self.l_real + self.l_fake + self.lambda_cls*self.c_real

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

        self.trainerG = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=0.5, beta2=0.9).minimize(
            self.g_loss, var_list=g_vars)

        self.trainerD = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=0.5, beta2=0.9).minimize(
            self.d_loss, var_list=d_vars)

        self.saver = tf.train.Saver(tf.global_variables())

