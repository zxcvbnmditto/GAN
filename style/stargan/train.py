import numpy as np
import tensorflow as tf
import utils
import os

from stargan import STARGAN

# Define Global params
tf.app.flags.DEFINE_integer('start_epoch', 0, 'Number of epoch to start running')
tf.app.flags.DEFINE_integer('epochs', 50000, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('d_iter', 5, 'Number of iterations training discriminator per epoch')
tf.app.flags.DEFINE_integer('g_iter', 1, 'Number of iterations training generator per epoch')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 100, 'Number of steps to save a ckpt')
tf.app.flags.DEFINE_integer('step_per_image', 1, 'Number of steps to save an image')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of latent')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Size of Embedding')
tf.app.flags.DEFINE_integer('sample_img_size', 12, 'Size of image')
tf.app.flags.DEFINE_integer('g_res_block', 8, 'Number of g_res_block')

tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')

tf.app.flags.DEFINE_list('using_attributes', ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'], 'Using Attributes')

tf.app.flags.DEFINE_string('model_dir', 'models/', 'Model path')
tf.app.flags.DEFINE_string('checkpoint_filename', 'stargan.ckpt', 'Checkpoint filename')

FLAGS = tf.app.flags.FLAGS

ALL_ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
              'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
              'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

USING_ATTIBUTES = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Male', 'Mustache', 'Young']

def main():
    # define data
    db = utils.DataLoader(FLAGS)

    # define model
    model = STARGAN(
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        latent_size=FLAGS.latent_size,
        img_size=FLAGS.sample_img_size,
        vocab_size=len(FLAGS.using_attributes),
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
        # summary_writer = tf.summary.FileWriter('logs/', graph=sess.graph)

        # training
        for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epochs):
            print('Epoch: {:d}\n'.format(epoch))

            # Update the discriminator
            for _ in range(FLAGS.d_iter):
                real_imgs, origin_tags, target_tags = db.get_nextbatch()
                feed_dict = {
                    model.real_imgs: real_imgs,
                    model.origin_tags: origin_tags,
                    model.target_tags: target_tags,
                    model.training: True
                }
                # _, summary_d, d_loss = sess.run([model.trainerD, model.d_sumop, model.d_loss], feed_dict=feed_dict)
                _, d_l, d_lr, d_lf, d_cr, d_grads = sess.run([model.trainerD, model.d_loss, model.l_real, model.l_fake, model.c_real, model.gradient_penalty], feed_dict=feed_dict)

            # Update the generator
            for _ in range(FLAGS.g_iter):
                real_imgs, origin_tags, target_tags = db.get_nextbatch()
                feed_dict = {
                    model.real_imgs: real_imgs,
                    model.origin_tags: origin_tags,
                    model.target_tags: target_tags,
                    model.training: True
                }
                # _, summary_g, g_loss = sess.run([model.trainerG, model.g_sumop, model.g_loss], feed_dict=feed_dict)
                _, g_l, g_lf, g_lrecon, g_cf = sess.run([model.trainerG, model.g_loss, model.l_fake, model.l_recon, model.c_fake], feed_dict=feed_dict)

            # Write loss to tensorboard
            # summary_writer.add_summary(summary_d, epoch)
            # summary_writer.add_summary(summary_g, epoch)
            # summary_writer.add_summary(summary_lr, epoch)
            # summary_writer.add_summary(summary_lw, epoch)
            # summary_writer.add_summary(summary_cr, epoch)
            # summary_writer.add_summary(summary_gf, epoch)
            # summary_writer.add_summary(summary_cf, epoch)

            print("Discriminator Loss: {:.3f} Generator Loss: {:.3f}\n".format(d_l, g_l))
            print("Discriminator Logit Loss: {:.3f} Discriminator Label Loss: {:.3f} Discriminator Gradient Loss: {:.3f}\n".format(-d_lr, d_cr, d_grads))
            print("Generator Logit Loss: {:.3f} Generator Label Loss: {:.3f} Generator Recon Loss: {:.3f}\n".format(-g_lf, g_cf, g_lrecon))

            # Graph the images
            if epoch % FLAGS.step_per_image == 0:
                feed_dict = {
                    model.real_imgs: db.test_imgs,
                    model.target_tags: db.test_tags,
                    model.training: False
                }
                f_imgs = sess.run([model.fake_imgs], feed_dict=feed_dict)
                utils.immerge_save(f_imgs, 1, FLAGS.sample_img_size, len(FLAGS.using_attributes))
            #
            # # Save model
            # if epoch % FLAGS.step_per_checkpoints == 0:
            #     ckpt_file = os.path.join(FLAGS.model_dir, FLAGS.checkpoint_filename)
            #     model.saver.save(sess, ckpt_file, global_step=epoch)

if __name__ == "__main__":
    main()
