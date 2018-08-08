import tensorflow as tf
import numpy as np
import os, glob

from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Reshape, UpSampling2D
from keras.optimizers import Adam

from skimage import io
from tqdm import tqdm
from PIL import Image

# Define Global params
tf.app.flags.DEFINE_integer('epochs', 200, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 100, 'Number of steps to save')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_integer('latent_size', 100, 'Size of Latent')
tf.app.flags.DEFINE_integer('model_num', 11, 'Number of old model to load')

tf.app.flags.DEFINE_bool('load', True, 'Load old model or not')
tf.app.flags.DEFINE_bool('train', True, 'Train model or not')

tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')

tf.app.flags.DEFINE_string('model_dir', 'model/', 'Model path')
tf.app.flags.DEFINE_string('checkpoint_filename', 'vanilla_gan.ckpt', 'Checkpoint filename')
FLAGS = tf.app.flags.FLAGS

data_dir = '../faces/'

class Vanilla_Gan():
    def __init__(self, learning_rate, batch_size, latent_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_dim = latent_size

        optimizer = Adam(lr=self.learning_rate, beta_1=0.5)

        # Build D
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build G
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)


        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


        self.generator.summary()
        self.discriminator.summary()
        self.combined.summary()


    def build_discriminator(self):
        print("BUILDING DISCRIMINATOR .....................")
        in_shape = (96, 96, 3)

        model = Sequential()
        model.add(Convolution2D(input_shape=in_shape, filters=48, kernel_size=(4, 4), strides=(2, 2), padding='SAME'))
        model.add(Activation('relu'))
        model.add(Convolution2D(filters=96, kernel_size=(4, 4), strides=(2, 2), padding='SAME'))
        model.add(Activation('relu'))
        model.add(Convolution2D(filters=192, kernel_size=(4, 4), strides=(2, 2), padding='SAME'))
        model.add(Activation('relu'))
        model.add(Convolution2D(filters=394, kernel_size=(4, 4), strides=(2, 2), padding='SAME'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(units=1,))
        model.add(Activation('sigmoid'))

        input = Input(shape=in_shape)
        output = model(input)

        model.summary()

        return Model(inputs=input, outputs=output)

    def build_generator(self):
        print("BUILDING GENERATOR .....................")
        in_shape = (self.latent_dim,)

        model = Sequential()
        model.add(Dense(input_shape=in_shape, units=48*24*24))
        model.add(Activation('relu'))
        model.add(Reshape((24, 24, 48)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(filters=192, kernel_size=(4, 4), strides=(1, 1), padding='SAME'))
        model.add(Activation('relu'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(filters=96, kernel_size=(4, 4), strides=(1, 1), padding='SAME'))
        model.add(Activation('relu'))
        model.add(Convolution2D(filters=3, kernel_size=(4, 4), strides=(1, 1), padding='SAME'))
        model.add(Activation('tanh'))

        input = Input(shape=in_shape)
        output = model(input)

        model.summary()

        return Model(inputs=input, outputs=output)


    def load_data(self):
        print("LOADING IMAGES ........")
        files = sorted(glob.glob(data_dir + '*.jpg'))

        images = []
        for file in files:
            image = io.imread(file)
            image = np.array(image)
            images.append(image)

        # [batch, 96, 96, 3]
        images = np.array(images)
        return images

    def create_batches(self, data):
        print("CREATING BATCHES ........")

        np.random.shuffle(data)

        batches = []
        for i in range(0, len(data), FLAGS.batch_size):
            if len(data) < i + FLAGS.batch_size:
                break
            batches.append(data[i: i + FLAGS.batch_size])

        batches = np.array(batches) / 255.0
        return batches

    def train(self):
        # Load the dataset
        images = self.load_data()

        if FLAGS.load:
            print("Loading Old Models ........................")
            self.generator = load_model('models/generator/generator-{:d}.h5'.format(FLAGS.model_num))
            self.discriminator = load_model('models/discriminator/discriminator-{:d}.h5'.format(FLAGS.model_num))
            self.combined = load_model('models/combined/combined-{:d}.h5'.format(FLAGS.model_num))

        # Adversarial ground truths
        real_labels = np.ones((self.batch_size, 1))
        fake_labels = np.zeros((self.batch_size, 1))

        for epoch in range(FLAGS.epochs):

            batches = self.create_batches(images)

            for batch in tqdm(batches):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Generate a batch of fake images
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the d_loss
                d_loss_real = self.discriminator.train_on_batch(batch, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Not sure if noise should be resampled
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.combined.train_on_batch(noise, real_labels)

                # Plot the progress
                print("[D loss: %f, acc.: %.2f%%] [G loss: %f]" % (d_loss[0], 100*d_loss[1], g_loss))

            self.generator.save('models/generator/generator-{:d}.h5'.format(epoch))
            self.discriminator.save('models/discriminator/discriminator-{:d}.h5'.format(epoch))
            self.combined.save('models/combined/combined-{:d}.h5'.format(epoch))

    def test(self):
        print("Loading Old Models ........................")
        self.generator = load_model('models/generator/generator-{:d}.h5'.format(FLAGS.model_num))

        noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
        outputs = self.generator.predict(noise)

        imgs = []
        for output in outputs:
            img = Image.fromarray( (output + 1) * 127 , 'RGB')
            # img.show()
            imgs.append(img)


if __name__ == "__main__":
    gan = Vanilla_Gan(FLAGS.learning_rate, FLAGS.batch_size, FLAGS.latent_size)

    if FLAGS.train:
        gan.train()
    else:
        gan.test()