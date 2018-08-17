import matplotlib.pyplot as plt
import numpy as np
import scipy
import glob
import os
import tensorflow as tf

from skimage import io

data_dir = '../faces/64-64/'

class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.batch_iter = -1
        self.c_dim = 3
        self.images = self.load_data()
        self.batches = self.create_new_batches()

    def load_data(self):
        print("LOADING IMAGES ........")
        files = sorted(glob.glob(data_dir + '*.jpg'))

        # adapted from https://github.com/changwoolee/WGAN-GP-tensorflow/blob/master/model.py
        # reader = tf.WholeFileReader()
        # filename_queue = tf.train.string_input_producer(files)
        # key, value = reader.read(filename_queue)
        # images = tf.image.decode_jpeg(value, channels=self.c_dim, name="dataset_image")
        #
        # images = tf.to_float(
        #     tf.image.resize_images(images, [64, 64], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
        #
        # batch = tf.train.shuffle_batch([images], batch_size=self.batch_size, capacity=30000,
        #                                min_after_dequeue=5000,
        #                                num_threads=4)

        images = []
        for file in files:
            image = io.imread(file)
            image = np.array(image)
            images.append(image)

        # [batch, 64, 64, 3]
        images = np.array(images)
        outputs = (images / 127.5) - 1
        return outputs

    def create_new_batches(self):
        print("CREATING NEW BATCHES ........")
        np.random.shuffle(self.images)

        batches = []
        for i in range(0, len(self.images), self.batch_size):
            if len(self.images) < i + self.batch_size:
                break
            batches.append(self.images[i: i + self.batch_size])

        batches = np.array(batches)
        return np.array(batches)

    def get_nextbatch(self):
        if self.batches.shape[0] <= self.batch_iter + 1:
            self.batches = self.create_new_batches()
            self.batch_iter = -1

        self.batch_iter += 1
        return self.batches[self.batch_iter]

def to_range(images, max, min, type):
    return ((images + 1.) / 2. * (max - min) + min).astype(type)


def immerge_save(images, epoch, img_size):
    images = np.array(images).squeeze()

    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    path_dir = 'sample_img/'

    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)

    path_dir = path_dir + str(epoch) + '.jpg'
    img = np.zeros((h * img_size, w * img_size, c))

    for idx, image in enumerate(images):
        i = idx % img_size
        j = idx // img_size
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    img = to_range(img, 0, 255, np.uint8)

    return io.imsave(path_dir, img)
