import numpy as np
import glob
import os
import csv
import random

from skimage import io
from scipy import misc

data_dirs = ['../faces/extra/']
tag_files = ['../faces/tags/tags.csv']
testing_tag_file = ['../faces/testing_tags/test_tags.csv']


class DataLoader:
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.sample_img_size = FLAGS.sample_img_size
        self.batch_iter = -1
        self.c_dim = 3
        self.color = []
        self.images = self.load_images()
        self.tags = self.gen_one_hot(self.load_tags(tag_files))
        self.create_testing_tags()
        self.test_tags = self.gen_one_hot(self.load_tags(testing_tag_file))


    def load_images(self):
        print("LOADING IMAGES ........")
        images = []
        for data_dir in data_dirs:
            files = sorted(glob.glob(data_dir + '*.jpg'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            for file in files:
                image = io.imread(file)
                image = np.array(image)
                images.append(image)

        images = np.array(images)
        print(images.shape)
        return images

    def load_tags(self, file_collections):
        print("LOADING Tags ........")
        tags = []
        for tag_file in file_collections:
            with open(tag_file, 'r') as f:
                rows = csv.reader(f)

                for row in rows:
                    hair_color = row[1].split()[0]
                    eye_color = row[1].split()[2]

                    if hair_color not in self.color:
                        self.color.append(hair_color)
                    if eye_color not in self.color:
                        self.color.append(eye_color)

                    tags.append([self.color.index(hair_color), self.color.index(eye_color)])

        outputs = np.array(tags)
        return outputs

    def gen_one_hot(self, inputs):
        print("Transforming Tag Pairs into One Hot Encoded structure ........")
        one_hot_vecs = []
        for tag in inputs:
            hair_vec = np.zeros(len(self.color))
            hair_vec[tag[0]] = 1
            eye_vec = np.zeros(len(self.color))
            eye_vec[tag[1]] = 1
            pair_vec = np.concatenate((hair_vec, eye_vec), axis=0)
            one_hot_vecs.append(pair_vec)

        outputs = np.array(one_hot_vecs)
        return outputs

    def get_nextbatch(self):
        # shuffle the batches
        if self.images.shape[0] <= (self.batch_iter + 2) * self.batch_size:
            rand_idx = np.random.permutation(len(self.images))

            self.images = self.images[rand_idx]
            self.tags = self.tags[rand_idx]
            self.batch_iter = -1

        self.batch_iter += 1

        # prepare wrong_imgs and wrong_tags
        rand_idx = np.random.permutation(len(self.images))
        rand_idx = rand_idx[:self.batch_size]
        wrong_imgs = self.images[rand_idx]
        rand_idx = np.random.permutation(len(self.images))
        rand_idx = rand_idx[:self.batch_size]
        wrong_tags = self.tags[rand_idx]

        return self.images[self.batch_iter * self.batch_size: (self.batch_iter + 1) * self.batch_size], \
               self.tags[self.batch_iter * self.batch_size: (self.batch_iter + 1) * self.batch_size], \
               wrong_imgs, wrong_tags

    def create_testing_tags(self):
        print("Creating Testing Tags ........")
        for file in testing_tag_file:
            f = open(file, "w")

            counter = 1
            for i in range(self.sample_img_size):
                tag = random.sample(self.color, 2)
                for _ in range(self.sample_img_size):
                    string = str(counter) + ',' + tag[0] + ' hair ' + tag[1] + ' eyes\n'
                    f.write(string)
                    counter += 1

            f.close()


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

    img = (img + 1.) / 2

    return misc.imsave(path_dir, img)


