import os
import utils
import numpy as np
import tensorflow as tf

from model import WGAN

# Define Global params
tf.app.flags.DEFINE_integer('epochs', 100000, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 100, 'Number of steps to save')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of latent')
tf.app.flags.DEFINE_integer('sample_img_size', 12, 'Size of image')


tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')

tf.app.flags.DEFINE_string('model_dir', 'models/', 'Model path')
tf.app.flags.DEFINE_string('checkpoint_filename', 'wgan-gp.ckpt', 'Checkpoint filename')

FLAGS = tf.app.flags.FLAGS

data_dir = '../faces/64-64/'


def main():
    # define data
    db = utils.DataLoader(FLAGS.batch_size)

    # define model here
    model = WGAN(
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        latent_size=FLAGS.latent_size,
        img_size=FLAGS.sample_img_size
    )

    with tf.Session() as sess:
        # load old model or not
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        # add summary writer for tensorboard
        summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)

        d_iters = 5
        # training
        for epoch in range(FLAGS.epochs):
            print('{:d}\n'.format(epoch))
            # Update the discriminator
            for _ in range(d_iters):
                # get the gaussian noise distribution
                z_batch = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
                x_batch = db.get_nextbatch()
                _, summary_d, summary_p = sess.run([model.trainerD, model.d_sumop, model.p_sumop],
                                                   feed_dict={model.Z: z_batch, model.X: x_batch, model.training: True})

            summary_writer.add_summary(summary_d, epoch)
            summary_writer.add_summary(summary_p, epoch)

            # Update the generator
            z_batch = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
            _, summary_g = sess.run([model.trainerG, model.g_sumop], feed_dict={model.Z: z_batch, model.training: True})
            summary_writer.add_summary(summary_g, epoch)

            # Show lost and draw the sampled fake images
            if epoch % 50 == 0:
                z_batch = np.random.normal(-1, 1, size=[FLAGS.sample_img_size ** 2, FLAGS.latent_size])
                f_imgs = sess.run([model.Gz], feed_dict={model.Z: z_batch, model.training: False})
                utils.immerge_save(f_imgs, epoch, FLAGS.sample_img_size)

            # save when the epoch % step_per_checkpoint == 0
            if epoch % FLAGS.step_per_checkpoints == 0:
                ckpt_file = os.path.join(FLAGS.model_dir, FLAGS.checkpoint_filename)
                model.saver.save(sess, ckpt_file, global_step=epoch)


if __name__ == "__main__":
    main()
