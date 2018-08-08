import glob
import os
import numpy as np
import tensorflow as tf

from skimage import io
from tqdm import tqdm
from wgan.wgan import WGAN
from PIL import Image

# Define Global params
tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of latent')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')

tf.app.flags.DEFINE_string('model_dir', 'wgan/models/', 'Model path')


FLAGS = tf.app.flags.FLAGS

data_dir = 'faces/64-64'

def main():
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

        # get the gaussian noise distribution
        z_batch = np.random.normal(-1, 1, size=[FLAGS.batch_size, 100])

        # Update the discriminator
        outputs = sess.run([model.Gz, ], feed_dict={model.Z: z_batch})

        outputs = np.array(outputs)
        outputs = np.squeeze(outputs)
        print(((outputs + 1) * 127.5))
        print(outputs.shape)

        imgs = []
        for output in outputs:
            img = Image.fromarray((output + 1) * 127.5, 'RGB')
            imgs.append(img)

        imgs[0].show()
        imgs[1].show()
        imgs[2].show()


if __name__ == "__main__":
    main()