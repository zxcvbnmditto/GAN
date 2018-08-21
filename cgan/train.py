import os
import utils
import numpy as np
import tensorflow as tf

from skimage import io

from model import CGAN

# Define Global params
tf.app.flags.DEFINE_integer('start_epoch', 0, 'Number of epoch to start running')
tf.app.flags.DEFINE_integer('epochs', 100, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 100, 'Number of steps to save')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of latent')
tf.app.flags.DEFINE_integer('sample_img_size', 12, 'Size of image')


tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')

tf.app.flags.DEFINE_string('model_dir', 'models/', 'Model path')
tf.app.flags.DEFINE_string('checkpoint_filename', 'cgan.ckpt', 'Checkpoint filename')

FLAGS = tf.app.flags.FLAGS

def main():
    # define data
    db = utils.DataLoader(FLAGS.batch_size)

    # define model here
    # model = CGAN(
    #     learning_rate=FLAGS.learning_rate,
    #     batch_size=FLAGS.batch_size,
    #     latent_size=FLAGS.latent_size,
    #     img_size=FLAGS.sample_img_size
    # )
    #
    # with tf.Session() as sess:
    #     # load old model or not
    #     ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    #     if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    #         print('Reloading model parameters..')
    #         model.saver.restore(sess, ckpt.model_checkpoint_path)
    #     else:
    #         print('Created new model parameters..')
    #         sess.run(tf.global_variables_initializer())
    #
    #     # add summary writer for tensorboard
    #     summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    #
    #     d_iters = 5
    #     # training
    #     for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epochs):
    #         print('{:d}\n'.format(epoch))
    #
    #         # save when the epoch % step_per_checkpoint == 0
    #         if epoch % FLAGS.step_per_checkpoints == 0:
    #             ckpt_file = os.path.join(FLAGS.model_dir, FLAGS.checkpoint_filename)
    #             model.saver.save(sess, ckpt_file, global_step=epoch)


if __name__ == "__main__":
    main()
