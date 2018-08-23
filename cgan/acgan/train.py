import numpy as np
import tensorflow as tf
import utils
import os

from acgan import ACGAN

# Define Global params
tf.app.flags.DEFINE_integer('start_epoch', 0, 'Number of epoch to start running')
tf.app.flags.DEFINE_integer('epochs', 50000, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('d_iter', 5, 'Number of iterations training discriminator per epoch')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 100, 'Number of steps to save a ckpt')
tf.app.flags.DEFINE_integer('step_per_image', 1, 'Number of steps to save an image')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of latent')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Size of Embedding')
tf.app.flags.DEFINE_integer('sample_img_size', 12, 'Size of image')

tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')

tf.app.flags.DEFINE_string('model_dir', 'models/', 'Model path')
tf.app.flags.DEFINE_string('checkpoint_filename', 'cgan.ckpt', 'Checkpoint filename')

FLAGS = tf.app.flags.FLAGS

def main():
    # define data
    db = utils.DataLoader(FLAGS)
    # define model
    model = ACGAN(
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        latent_size=FLAGS.latent_size,
        img_size=FLAGS.sample_img_size,
        vocab_size=len(db.color)
    )

    # print(db.color)

    with tf.Session() as sess:
        # load old model or not
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        # Create Tensorboard
        # summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)

        # training
        for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epochs):
            # print('{:d}\n'.format(epoch))

            # Update the discriminator
            for _ in range(FLAGS.d_iter):
                noise = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
                c_imgs, c_tags, w_imgs, w_tags = db.get_nextbatch()
                feed_dict = {
                    model.correct_imgs: c_imgs,
                    model.correct_tags: c_tags,
                    # model.wrong_imgs: w_imgs,
                    # model.wrong_tags: w_tags,
                    model.noise: noise,
                    model.training: True
                }

                _, summary_d, c_loss, s_loss, temp3 = sess.run(
                    [model.trainerD, model.d_sumop, model.loss_condition, model.loss_source, model.temp3], feed_dict=feed_dict)

                # temp3 = ['{:.2f}'.format(i) for i in temp3]

                print('\ncEpoch: {:d} C_loss: {:f} S_loss: {:f}\n'.format(epoch, c_loss, s_loss))
                print(temp3.shape)

            # Update the generator
            noise = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
            c_imgs, c_tags, w_imgs, w_tags = db.get_nextbatch()
            feed_dict = {
                model.correct_imgs: c_imgs,
                model.correct_tags: c_tags,
                # model.wrong_imgs: w_imgs,
                # model.wrong_tags: w_tags,
                model.noise: noise,
                model.training: True
            }
            _, summary_g = sess.run([model.trainerG, model.g_sumop], feed_dict=feed_dict)

            # Write loss to tensorboard
            # summary_writer.add_summary(summary_d, epoch)
            # summary_writer.add_summary(summary_g, epoch)

            # Graph the images
            if epoch % FLAGS.step_per_image == 0:
                noise = np.random.normal(-1, 1, size=[FLAGS.sample_img_size ** 2, FLAGS.latent_size])
                # the correct_tags here should be generated from testing tag file
                feed_dict = {
                    model.correct_tags: db.test_tags,
                    model.noise: noise,
                    model.training: False
                }
                f_imgs = sess.run([model.fake_imgs], feed_dict=feed_dict)
                utils.immerge_save(f_imgs, 1, FLAGS.sample_img_size)

            # Save model
            # if epoch % FLAGS.step_per_checkpoints == 0:
            #     ckpt_file = os.path.join(FLAGS.model_dir, FLAGS.checkpoint_filename)
            #     model.saver.save(sess, ckpt_file, global_step=epoch)


if __name__ == "__main__":
    main()
