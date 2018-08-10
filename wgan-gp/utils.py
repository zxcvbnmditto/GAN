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
        self.images = self.load_data()
        self.batch_iter = -1
        self.batch_size = batch_size
        self.batches = self.create_new_batches()

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


def immerge_save(images, epoch):
    images = np.array(images).squeeze()

    row, col = 16, 16
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    path_dir = 'sample_img/'

    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)

    path_dir = path_dir + str(epoch) + '.jpg'
    img = np.zeros((h * row, w * col, c))

    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    img = to_range(img, 0, 255, np.uint8)

    return io.imsave(path_dir, img)
