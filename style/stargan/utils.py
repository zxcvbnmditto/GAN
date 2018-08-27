import numpy as np
import glob
import os
import csv
import random

from skimage import io
from scipy import misc

data_dirs = ['../../faces/celeb/128_128/']
tag_files = ['../../faces/celeb/tags/tags.txt']


class DataLoader:
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.sample_img_size = FLAGS.sample_img_size
        self.batch_iter = -1
        self.c_dim = 3
        self.temp_max_file = 50000
        self.all_attributes = []
        self.using_attributes = FLAGS.using_attributes
        self.images = self.load_images()
        self.tags = self.load_tags(tag_files)
        self.test_imgs = self.load_testing_imgs()
        self.test_tags = self.load_testing_tags()


    def load_images(self):
        print("LOADING IMAGES ........")
        images = []
        for data_dir in data_dirs:
            files = sorted(glob.glob(data_dir + '*.jpg'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            for id, file in enumerate(files):
                if id > self.temp_max_file:
                    break
                image = io.imread(file)
                image = np.array(image)
                images.append(image)

        outputs = np.array(images)
        print(outputs.shape)
        return outputs

    def load_tags(self, file_collections):
        print("LOADING Tags ........")
        tags = []
        for tag_file in file_collections:
            with open(tag_file, 'r') as f:

                for id, row in enumerate(f):
                    if id == 0:
                        continue
                    elif id == 1:
                        self.all_attributes = [a for a in row.split()]
                    elif id <= self.temp_max_file + 2:
                        tag = []
                        for a in self.using_attributes:
                            idx = self.all_attributes.index(a)
                            if (row.split())[idx + 1] == '-1':
                                tag.append(0)
                            else:
                                tag.append(1)

                        tags.append(tag)

        outputs = np.array(tags)
        print(outputs.shape)
        return outputs


    def get_nextbatch(self):
        # shuffle the batches
        if self.images.shape[0] <= (self.batch_iter + 2) * self.batch_size:
            rand_idx = np.random.permutation(len(self.images))

            self.images = self.images[rand_idx]
            self.tags = self.tags[rand_idx]
            self.batch_iter = -1

        self.batch_iter += 1

        target_tags = []
        idxs = np.random.randint(0, len(self.using_attributes) - 1, self.batch_size)
        for i, idx in enumerate(idxs):
            tag = np.zeros(len(self.using_attributes))
            tag[idx] = 1
            target_tags.append(tag)


        return self.images[self.batch_iter * self.batch_size: (self.batch_iter + 1) * self.batch_size], \
               self.tags[self.batch_iter * self.batch_size: (self.batch_iter + 1) * self.batch_size], \
                np.array(target_tags)

    def load_testing_imgs(self):
        print("Loading Testing Images ........")

        imgs = []
        index = np.random.randint(0, len(self.images), self.sample_img_size)
        for i in index:
            for _ in range(len(self.using_attributes)):
                imgs.append(self.images[i])

        outputs = np.array(imgs)
        print(outputs.shape)
        return outputs


    def load_testing_tags(self):
        print("Loading Testing Tags ........")

        testing_tags = []
        for i in range(self.sample_img_size):
            for j in range(len(self.using_attributes)):
                tag = np.zeros(len(self.using_attributes))
                tag[j] = 1
                testing_tags.append(tag)

        outputs = np.array(testing_tags)
        print(outputs.shape)
        return outputs

def add_sample_noise(inputs):
    return inputs + 0.5 * inputs.std() * np.random.random(inputs.shape)

def immerge_save(images, epoch, img_size, a_size):
    images = np.array(images).squeeze()

    # print(images.shape)

    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    path_dir = 'sample_img/'

    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)

    path_dir = path_dir + str(epoch) + '.jpg'
    img = np.zeros((h * a_size, w * img_size, c))

    for idx, image in enumerate(images):
        i = idx % img_size
        j = idx // img_size
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    img = (img + 1.) / 2

    return misc.imsave(path_dir, img)


