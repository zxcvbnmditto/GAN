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
        outputs = ops.batch_norm(outputs, training)
        outputs = tf.nn.relu(outputs)

    return outputs

# the actual element wise sum resblock
def resblock1_D(inputs, filters, kernel_size, stride, training, norm, collection, scope):
    with tf.variable_scope(scope):
        outputs = ops.conv_2d(inputs, filters, kernel_size, stride, padding='SAME', stddev=0.02, norm=norm, training=training, collection=collection, scope='conv_1')
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)
        outputs = ops.conv_2d(outputs, filters, kernel_size, stride, padding='SAME', stddev=0.02, norm=norm, training=training, collection=collection, scope='conv_2')
        outputs = outputs + inputs
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)

    return outputs

# for sizing up
def resblock2_D(inputs, filters, kernel_size, stride, training, norm, collection, scope):
    with tf.variable_scope(scope):
        outputs = ops.conv_2d(inputs, filters, kernel_size, stride, padding='SAME', stddev=0.02, norm=norm, training=training, collection=collection, scope='conv')
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)

    return outputs


class DRAGAN():
    def __init__(self, FLAGS):
        print("Constructing DRAGAN model ........")
        self.batch_size = FLAGS.fixed_batch_size
        self.latent_size = FLAGS.latent_size
        self.feature_length = FLAGS.feature_length
        self.res_block_size = FLAGS.res_block_size
        self.d_norm = "layer_norm"
        self.g_norm = "batch_norm"
        self.lambda_adv = self.feature_length
        self.lambda_gp = 5

        self.d_lr = tf.placeholder(tf.float32)
        self.g_lr = tf.placeholder(tf.float32)
        self.noise = tf.placeholder(tf.float32, shape=[None, self.latent_size])
        # correct imgs
        self.c_imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        # correct_tags
        self.c_tags = tf.placeholder(tf.float32, shape=[None, self.feature_length])

        self.training = tf.placeholder(tf.bool)
        self.build()

    def discriminator(self, x, reuse=False, collection=None):
        print("Discriminator ...........")
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                scope.reuse_variables()

            inputs = x

            outputs = resblock2_D(inputs, 32, 4, 2, self.training, self.d_norm, collection, scope='d_head_0')
            for i in range(2):
                outputs = resblock1_D(outputs, 32, 3, 1, self.training, self.d_norm, collection, scope='d_body_0_{:d}'.format(i))

            outputs = resblock2_D(outputs, 64, 4, 2, self.training, self.d_norm, collection, scope='d_head_1')
            for i in range(2):
                outputs = resblock1_D(outputs, 64, 3, 1, self.training, self.d_norm, collection, scope='d_body_1_{:d}'.format(i))

            outputs = resblock2_D(outputs, 128, 4, 2, self.training, self.d_norm, collection, scope='d_head_2')
            for i in range(2):
                outputs = resblock1_D(outputs, 128, 3, 1, self.training, self.d_norm, collection, scope='d_body_2_{:d}'.format(i))

            outputs = resblock2_D(outputs, 256, 4, 2, self.training, self.d_norm, collection, scope='d_head_3')
            for i in range(2):
                outputs = resblock1_D(outputs, 256, 3, 1, self.training, self.d_norm, collection, scope='d_body_3_{:d}'.format(i))

            outputs = resblock2_D(outputs, 512, 4, 2, self.training, self.d_norm, collection, scope='d_head_4')
            for i in range(2):
                outputs = resblock1_D(outputs, 512, 3, 1, self.training, self.d_norm, collection, scope='d_body_4_{:d}'.format(i))

            outputs = resblock2_D(outputs, 1024, 4, 2, self.training, self.d_norm, collection, scope='d_head_5')
            outputs = tf.reshape(outputs, [-1, 1 * 1 * 1024])

            logits = ops.dense(outputs, 1, 0.02, self.training, None, collection, scope="d_logits")

            labels = ops.dense(outputs, self.feature_length, 0.02, self.training, None, collection, scope="d_labels")

        return logits, labels

    def generator(self, x, c, reuse=False):
        print("Generator ...........")

        with tf.variable_scope('generator') as scope:
            if (reuse):
                scope.reuse_variables()

            inputs = tf.concat([x, c], axis=-1)

            # Dense layer 1
            dense1 = ops.dense(inputs, 64*8*8, 0.02, self.training, norm=self.g_norm, scope="g_inputs")
            dense1 = tf.nn.relu(dense1)
            dense1 = tf.reshape(dense1, [-1, 8, 8, 64])

            outputs = dense1
            # Res Block
            for i in range(self.res_block_size):
                outputs = resblock1_G(outputs, 64, 3, 1, self.training, norm=self.g_norm, scope='g_residual_{:d}'.format(i))

            outputs = ops.batch_norm(outputs, self.training)
            outputs = tf.nn.relu(outputs)
            outputs = outputs + dense1

            # Upscaling by pixel shuffling
            for i in range(3):
                outputs = resblock2_G(outputs, 256, 3, 1, 2, self.training, norm=None, scope='g_upscale_{:d}'.format(i))

            outputs = ops.conv_2d(outputs, 3, 9, 1, padding='SAME', stddev=0.02, training=self.training, norm=None, scope="g_conv_last")
            outputs = tf.nn.tanh(outputs)

        return outputs

    def build(self):
        # fake imgs => sampled by pc based on correct tags
        self.fake_imgs = self.generator(self.noise, self.c_tags, reuse=False)

        # correct logits and labels => results of pc's analysis given imgs from real database
        self.c_logits, self.c_labels = self.discriminator(self.c_imgs, reuse=False, collection=None)
        # fake logits and labels => results of pc's analysis given imgs generated by the generator using correct tags
        self.f_logits, self.f_labels = self.discriminator(self.fake_imgs, reuse=True, collection='NO_OPS')

        # DRAGAN
        # g_fake => used for evaluate generator's logits loss
        # closed to one is better
        self.g_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logits, labels=tf.ones_like(self.f_logits)))
        # self.g_fake = -tf.reduce_mean(self.f_logits)

        # Logit Loss
        # l_real => used for evaluate discriminator's logit loss on correct logits
        # closed to one is better
        self.l_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.c_logits, labels=tf.ones_like(self.c_logits)))
        # self.l_real = -tf.reduce_mean(self.c_logits)

        # l_fake => used for evaluate discriminator's logit loss on fake logits
        # closed to zero is better
        self.l_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_logits, labels=tf.zeros_like(self.f_logits)))
        # self.l_fake = tf.reduce_mean(self.f_logits)

        # Label Loss
        # c_real => used for evaluate discriminaor's label loss given correct image from based on correct tags
        self.c_real = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.c_labels, labels=self.c_tags), axis=1))
        # c_fake => used for evaluate discriminaor's label loss given fake image produce by pc based on correct labels
        # want it to fool the fake label to be as close as correct tags => fool the discriminator => better img potential
        self.c_fake = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.f_labels, labels=self.c_tags), axis=1))

        # Penalty Loss
        penalty_dist = tf.random_uniform(shape=[self.batch_size, 1], minval=0, maxval=1)
        differences = self.fake_imgs - self.c_imgs
        interpolates = self.c_imgs + penalty_dist * differences
        interpolates_logits, _ = self.discriminator(interpolates, reuse=True)
        grads = tf.gradients(interpolates_logits, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

        self.g_loss = self.lambda_adv*self.g_fake + self.c_fake
        self.d_loss = self.lambda_adv*self.l_real + self.c_real + self.lambda_adv*self.l_fake + self.c_fake
        # self.d_loss = (self.lambda_adv / 2)*self.l_real + self.c_real + (self.lambda_adv / 2)*self.l_fake


        self.d_sumop = tf.summary.scalar('d_loss', self.d_loss)
        self.g_sumop = tf.summary.scalar('g_loss', self.g_loss)
        self.p_sumop = tf.summary.scalar('p_loss', self.gradient_penalty)


        # self.lr_sumop = tf.summary.scalar('l_real', self.l_real)
        # self.lw_sumop = tf.summary.scalar('l_wrong', self.l_wrong)
        # self.cr_sumop = tf.summary.scalar('c_real', self.c_real)
        # self.gf_sumop = tf.summary.scalar('g_fake', self.g_fake)
        # self.cf_sumop = tf.summary.scalar('c_fake', self.c_fake)

        d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # update update_ops before processing the D/G trainer
        with tf.control_dependencies(update_ops):
            self.trainerG = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=0.5, beta2=0.9).minimize(
                self.g_loss, var_list=g_vars)

            self.trainerD = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=0.5, beta2=0.9).minimize(
                self.d_loss, var_list=d_vars)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

