import numpy as np
import tensorflow as tf
import utils
import os

from dragan import DRAGAN

# Define Global params
tf.app.flags.DEFINE_integer('start_epoch', 7000, 'Number of epoch to start running')
tf.app.flags.DEFINE_integer('epochs', 75000, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('d_iter', 1, 'Number of iterations training discriminator per epoch')
tf.app.flags.DEFINE_integer('g_iter', 1, 'Number of iterations training generator per epoch')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 100, 'Number of steps to save a ckpt')
tf.app.flags.DEFINE_integer('step_per_image', 100, 'Number of steps to save an image')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of latent')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Size of Embedding')
tf.app.flags.DEFINE_integer('sample_img_size', 12, 'Size of image')
tf.app.flags.DEFINE_integer('g_res_block', 16, 'Number of g_res_block')


tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')

tf.app.flags.DEFINE_string('model_dir', 'models/', 'Model path')
tf.app.flags.DEFINE_string('checkpoint_filename', 'cgan.ckpt', 'Checkpoint filename')

FLAGS = tf.app.flags.FLAGS

def main():
    # define data
    db = utils.DataLoader(FLAGS)

    print(db.hair_color_dict)
    print(db.eye_color_dict)

    # define model
    model = DRAGAN(
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        latent_size=FLAGS.latent_size,
        img_size=FLAGS.sample_img_size,
        vocab_size=len(db.hair_color_dict) + len(db.eye_color_dict),
        res_block_size=FLAGS.g_res_block
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

        # Create Tensorboard
        summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)

        # training
        for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epochs):
            print('Epoch: {:d}\n'.format(epoch))

            # Update the discriminator
            for _ in range(FLAGS.d_iter):
                noise = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
                c_imgs, c_tags, r_imgs, r_tags = db.get_nextbatch()
                c_p = utils.add_sample_noise(c_imgs)
                feed_dict = {
                    model.c_imgs: c_imgs,
                    model.c_tags: c_tags,
                    model.p_imgs: c_p,
                    model.r_imgs: r_imgs,
                    model.r_tags: r_tags,
                    model.noise: noise,
                    model.training: True
                }
                _, summary_d, summary_lr, summary_lw, summary_cr, d_loss = sess.run([model.trainerD, model.d_sumop, model.lr_sumop, model.lw_sumop, model.cr_sumop, model.d_loss], feed_dict=feed_dict)
                # _, summary_d, d_logit_loss, d_label_loss, d_loss = sess.run([model.trainerD, model.d_sumop, model.l_real, model.c_real, model.d_loss], feed_dict=feed_dict)

            # Update the generator
            for _ in range(FLAGS.g_iter):
                noise = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
                c_imgs, c_tags, r_imgs, r_tags = db.get_nextbatch()
                c_p = utils.add_sample_noise(c_imgs)
                feed_dict = {
                    model.c_imgs: c_imgs,
                    model.c_tags: c_tags,
                    model.p_imgs: c_p,
                    model.r_imgs: r_imgs,
                    model.r_tags: r_tags,
                    model.noise: noise,
                    model.training: True
                }
                _, summary_g, summary_gf, summary_cf, g_loss = sess.run([model.trainerG, model.g_sumop, model.gf_sumop, model.cf_sumop, model.g_loss], feed_dict=feed_dict)
                # _, summary_g, g_logit_loss, g_label_loss, g_loss = sess.run([model.trainerG, model.g_sumop, model.g_fake, model.c_fake, model.g_loss], feed_dict=feed_dict)

            # Write loss to tensorboard
            summary_writer.add_summary(summary_d, epoch)
            summary_writer.add_summary(summary_g, epoch)
            summary_writer.add_summary(summary_lr, epoch)
            summary_writer.add_summary(summary_lw, epoch)
            summary_writer.add_summary(summary_cr, epoch)
            summary_writer.add_summary(summary_gf, epoch)
            summary_writer.add_summary(summary_cf, epoch)

            # print("Discriminator Loss: {:.3f} Generator Loss: {:.3f}\n".format(d_loss, g_loss))
            # print("Discriminator Logit Loss: {:.3f} Discriminator Label Loss: {:.3f}\n".format(d_logit_loss, d_label_loss))
            # print("Generator Logit Loss: {:.3f} Generator Label Loss: {:.3f}\n".format(g_logit_loss, g_label_loss))

            # Graph the images
            if epoch % FLAGS.step_per_image == 0:
                noise = np.random.normal(-1, 1, size=[FLAGS.sample_img_size ** 2, FLAGS.latent_size])
                # the correct_tags here should be generated from testing tag file
                feed_dict = {
                    model.c_tags: db.test_tags,
                    model.noise: noise,
                    model.training: False
                }
                f_imgs = sess.run([model.fake_imgs], feed_dict=feed_dict)
                utils.immerge_save(f_imgs, epoch, FLAGS.sample_img_size)

            # Save model
            if epoch % FLAGS.step_per_checkpoints == 0:
                ckpt_file = os.path.join(FLAGS.model_dir, FLAGS.checkpoint_filename)
                model.saver.save(sess, ckpt_file, global_step=epoch)


if __name__ == "__main__":
    main()
