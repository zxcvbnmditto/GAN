import glob
import os
import numpy as np
import tensorflow as tf

from skimage import io
from tqdm import tqdm
from wgan.wgan import WGAN

# Define Global params
tf.app.flags.DEFINE_integer('epochs', 20000, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 100, 'Number of steps to save')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of latent')

tf.app.flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate')

tf.app.flags.DEFINE_string('model_dir', 'wgan/models/', 'Model path')
tf.app.flags.DEFINE_string('checkpoint_filename', 'wgan.ckpt', 'Checkpoint filename')

FLAGS = tf.app.flags.FLAGS

data_dir = 'faces/64-64/'

def load_data():
    print("LOADING IMAGES ........")
    files = sorted(glob.glob(data_dir + '*.jpg'))

    images = []
    for file in files:
        image = io.imread(file)
        image = np.array(image)
        images.append(image)

    # [batch, 96, 96, 3]
    images = np.array(images)
    outputs = (np.array(images) - 127.5) / 127.5
    return outputs


def create_new_batches():
    print("CREATING NEW BATCHES ........")
    np.random.shuffle(IMAGES)

    batches = []
    for i in range(0, len(IMAGES), FLAGS.batch_size):
        if len(IMAGES) < i + FLAGS.batch_size:
            break
        batches.append(IMAGES[i: i + FLAGS.batch_size])

    batches = np.array(batches)
    return np.array(batches)


def get_nextbatch(batches, batch_iter):
    if batches.shape[0] <= batch_iter + 1:
        batches = create_new_batches()
        batch_iter = -1

    batch_iter += 1
    return batches[batch_iter], batch_iter, batches


def main():
    # load raw image data
    global IMAGES
    IMAGES = load_data()

    # shuffle data and create batches
    batches = create_new_batches()

    with tf.Session() as sess:
        # define model here
        model = WGAN(
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            latent_size=FLAGS.latent_size
        )

        # load old model or not
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        # add summary writer for tensorboard
        summary_writer = tf.summary.FileWriter('wgan/logs/', graph=sess.graph)

        batch_iter = -1
        # training
        for epoch in range(FLAGS.epochs):
            d_iters = 5
            # set d_iters more when first start training and every 500 epochs
            if epoch < 25 or epoch % 500 == 0:
                d_iters = 100

            # Update the discriminator
            for _ in range(d_iters):
                # get the gaussian noise distribution
                z_batch = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
                x_batch, batch_iter, batches = get_nextbatch(batches, batch_iter=batch_iter)

                sess.run([model.d_clip])
                sess.run([model.trainerD], feed_dict={model.Z: z_batch, model.X: x_batch})

            # Update the generator
            z_batch = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
            sess.run([model.trainerG], feed_dict={model.Z: z_batch})

            # Show lost

            if epoch < 25 or epoch % FLAGS.step_per_checkpoints == 0:
                z_batch = np.random.normal(-1, 1, size=[FLAGS.batch_size, FLAGS.latent_size])
                x_batch, batch_iter, batches = get_nextbatch(batches, batch_iter=batch_iter)

                dLoss = sess.run(model.d_loss, feed_dict={model.Z: z_batch, model.X: x_batch})
                gLoss = sess.run(model.g_loss, feed_dict={model.Z: z_batch})

                print(str(epoch) + "\n")
                print("Discriminator Loss:  {:.4f} .......... \n".format(dLoss))
                print("Generator Loss:  {:.4f} .......... \n".format(gLoss))

            # save when the epoch % step_per_checkpoint == 0
            if epoch % FLAGS.step_per_checkpoints == 0:
                ckpt_file = os.path.join(FLAGS.model_dir, FLAGS.checkpoint_filename)
                model.saver.save(sess, ckpt_file, global_step=epoch)

if __name__ == "__main__":
    main()