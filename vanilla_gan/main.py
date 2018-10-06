import os
import numpy as np
import tensorflow as tf
import time
import glob

from new_wgan_gp import WGAN_GP
import new_utils

# Define Global params
# Customize these once for each dataset
# tf.app.flags.DEFINE_string('model_dir', '/media/james/D/Datasets/anime_character/vanilla/models/first', 'Path to save model')
# tf.app.flags.DEFINE_string('logs_dir', '/media/james/D/Datasets/anime_character/vanilla/logs/first', 'Path to save tensorboard logging')
# tf.app.flags.DEFINE_string('img_dir', '/media/james/D/Datasets/anime_character/vanilla/sample_imgs/first', 'Path to save images')
# tf.app.flags.DEFINE_string('tfrecords_dir', '/media/james/D/Datasets/anime_character/tfrecords', 'Path to the tfrecord file')
# tf.app.flags.DEFINE_string('rawdata_dir', '/media/james/D/Datasets/anime_character/raw_data/96-96', 'Path to the raw images')
# tf.app.flags.DEFINE_string('checkpoint_filename', 'wgan-gp.ckpt', 'Checkpoint filename')
# tf.app.flags.DEFINE_string('tfrecords_filename', 'anime_character_no_tags.tfrecords', 'Tfrecord filename')

tf.app.flags.DEFINE_string('model_dir', '/media/james/D/Datasets/celeb/vanilla/models/first', 'Path to save model')
tf.app.flags.DEFINE_string('logs_dir', '/media/james/D/Datasets/celeb/vanilla/logs/first', 'Path to save tensorboard logging')
tf.app.flags.DEFINE_string('img_dir', '/media/james/D/Datasets/celeb/vanilla/sample_imgs/first', 'Path to save images')
tf.app.flags.DEFINE_string('tfrecords_dir', '/media/james/D/Datasets/celeb/tfrecords', 'Path to the tfrecord file')
tf.app.flags.DEFINE_string('rawdata_dir', '/media/james/D/Datasets/celeb/raw_data/178_218', 'Path to the raw images')
tf.app.flags.DEFINE_string('checkpoint_filename', 'wgan-gp.ckpt', 'Checkpoint filename')
tf.app.flags.DEFINE_string('tfrecords_filename', 'celeb_no_tags.tfrecords', 'Tfrecord filename')


# Less Likely to worry about
tf.app.flags.DEFINE_integer('fixed_batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of latent/random noise')
tf.app.flags.DEFINE_integer('sample_img_size', 12, 'Generate a testing image which is a combination of sample_img_size**2 imgs')
tf.app.flags.DEFINE_integer('d_iters', 5, 'Discrimintor iteration count per epoch')
tf.app.flags.DEFINE_integer('g_iters', 1, 'Generator iteration count per epoch')
tf.app.flags.DEFINE_integer('capacity', 7000, 'Capacity of the buffer')
tf.app.flags.DEFINE_integer('num_threads', 12, 'Number of threads to use')
tf.app.flags.DEFINE_integer('min_after_dequeue', 2000, 'Minium after dequeue')
tf.app.flags.DEFINE_integer('img_channel', 3, 'Number of channels of an input image')
tf.app.flags.DEFINE_integer('kernel_size', 5, 'Size of a filter, celeb -> 5, anime -> 5')
tf.app.flags.DEFINE_integer('min_img_size', 4, 'Minimum size of the feature map size, celeb -> 4, anime -> 4')

# Check everytime before running
tf.app.flags.DEFINE_bool('training', True, 'Train or Test')
tf.app.flags.DEFINE_bool('create_tfrecord', False, 'Create New tfrecord, required to be True for the first run')
tf.app.flags.DEFINE_float('gpu_fraction', 0.0, 'GPU fraction, 0.0 => allow growth')
tf.app.flags.DEFINE_integer('start_epoch', 0, 'Use 0 for new training, use n to continue from previous trails')
tf.app.flags.DEFINE_integer('epochs', 25000, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 50, 'Number of steps to save per checkpoint')
tf.app.flags.DEFINE_integer('step_per_image', 50, 'Number of steps to save per testing image')
tf.app.flags.DEFINE_integer('img_size', 128, 'Height of the output image, assume height == width')

tf.app.flags.DEFINE_float('d_lr', 0.0001, 'Discriminator Learning rate')
tf.app.flags.DEFINE_float('g_lr', 0.0001, 'Generator Learning rate')

FLAGS = tf.app.flags.FLAGS

rawdata_filenames = sorted(glob.glob(os.path.join(FLAGS.rawdata_dir, "*.jpg")),
                           key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

def create_filequeue(sess, epochs):
    filename_queue = tf.train.string_input_producer([os.path.join(FLAGS.tfrecords_dir, FLAGS.tfrecords_filename)],
                                                    shuffle=False, num_epochs=epochs)
    image_batch = new_utils.read_and_decode(filename_queue, FLAGS)
    sess.run(tf.local_variables_initializer())
    return image_batch

def train(config):
    # define model here
    model = WGAN_GP(FLAGS)

    with tf.Session(config=config) as sess:
        # load old model or create new model
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())
        # Tensorboard
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, graph=sess.graph)

        # Prepare Tensorflow dataloader
        batch_per_epoch = len(rawdata_filenames) / FLAGS.fixed_batch_size / FLAGS.d_iters
        num_epochs = FLAGS.epochs / batch_per_epoch + 1
        image_batch = create_filequeue(sess, num_epochs)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Fixed noises for testing images
        fixed_z_batch = np.random.normal(-1, 1, size=[FLAGS.sample_img_size ** 2, FLAGS.latent_size])

        # training
        for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epochs):
            start_time = time.time()
            # Update the discriminator
            for iter in range(FLAGS.d_iters):
                z_batch = np.random.normal(-1, 1, size=[FLAGS.fixed_batch_size, FLAGS.latent_size])
                x_batch = sess.run(image_batch)
                feed_dict = {model.Z: z_batch, model.X: x_batch, model.moving_batch_size: FLAGS.fixed_batch_size,
                             model.training: True, model.d_lr: FLAGS.d_lr}
                _, summary_d, summary_p, d_loss = sess.run([model.trainerD, model.d_sumop, model.p_sumop, model.d_loss],
                                                   feed_dict=feed_dict)

            # Update the generator
            for iter in range(FLAGS.g_iters):
                z_batch = np.random.normal(-1, 1, size=[FLAGS.fixed_batch_size, FLAGS.latent_size])
                feed_dict = {model.Z: z_batch, model.moving_batch_size: FLAGS.fixed_batch_size,
                             model.training: True, model.g_lr: FLAGS.g_lr}
                _, summary_g, g_loss = sess.run([model.trainerG, model.g_sumop, model.g_loss], feed_dict=feed_dict)

            # Save Summaries
            print('Epoch: {:d} D_Loss: {:f} G_Loss: {:f} Time: {:.3f}'.format(epoch, d_loss, g_loss, time.time()-start_time))
            summary_writer.add_summary(summary_d, epoch)
            summary_writer.add_summary(summary_p, epoch)
            summary_writer.add_summary(summary_g, epoch)

            # Plot Images
            if epoch % FLAGS.step_per_image == 0:
                feed_dict = {model.Z: fixed_z_batch, model.moving_batch_size: FLAGS.sample_img_size ** 2,
                             model.training: False}
                f_imgs = sess.run([model.Gz], feed_dict=feed_dict)

                path_dir = os.path.join(FLAGS.img_dir, 'training_results')
                if not os.path.isdir(path_dir): os.makedirs(path_dir)
                new_utils.immerge_save(f_imgs, epoch, FLAGS.sample_img_size, path_dir)

            # Save Model
            if epoch % FLAGS.step_per_checkpoints == 0:
                ckpt_file = os.path.join(FLAGS.model_dir, FLAGS.checkpoint_filename)
                model.saver.save(sess, ckpt_file, global_step=epoch)

        coord.request_stop()
        coord.join(threads)

def test(config):
    # define model here
    model = WGAN_GP(FLAGS)

    with tf.Session(config=config) as sess:
        # load old model or not
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        path_dir = os.path.join(FLAGS.img_dir, 'testing_results')
        if not os.path.isdir(path_dir): os.makedirs(path_dir)
        # Create arbitrary number (100) of sample_images
        for num in range(100):
            z_batch = np.random.normal(-1, 1, size=[FLAGS.sample_img_size ** 2, FLAGS.latent_size])
            feed_dict = {model.Z: z_batch, model.moving_batch_size: FLAGS.sample_img_size ** 2,
                         model.training: False}
            f_imgs = sess.run([model.Gz], feed_dict=feed_dict)

            new_utils.immerge_save(f_imgs, num, FLAGS.sample_img_size, path_dir)

if __name__ == "__main__":
    config = tf.ConfigProto()
    if FLAGS.gpu_fraction > 0.0:
        print("GPU Fraction: {:.3f}".format(FLAGS.gpu_fraction))
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    else:
        print("GPU Fraction: Allow Growth")
        config.gpu_options.allow_growth = True

    if not os.path.isdir(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    if not os.path.isdir(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    if not os.path.isdir(FLAGS.img_dir):
        os.makedirs(FLAGS.img_dir)

    if not os.path.isdir(FLAGS.tfrecords_dir):
        os.makedirs(FLAGS.tfrecords_dir)

    if FLAGS.create_tfrecord:
        new_utils.create_tfrecord(os.path.join(FLAGS.tfrecords_dir, FLAGS.tfrecords_filename), rawdata_filenames)

    if FLAGS.training:
        print("Training")
        train(config)
    else:
        print("Testing")
        test(config)