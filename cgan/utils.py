import numpy as np
import glob
import os
import csv

from skimage import io
from scipy import misc

data_dirs = ['../faces/extra/']
tag_files = ['../faces/tags/tags.csv']

class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.batch_iter = -1
        self.c_dim = 3
        self.images = self.load_images()
        self.tags = self.load_tags()

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

    def load_tags(self):
        print("LOADING Tags ........")
        tags = []
        for tag_file in tag_files:
            with open(tag_file, 'r') as f:
                rows = csv.reader(f)

                for row in rows:
                    hair_color = row[1].split()[0]
                    eye_color = row[1].split()[2]
                    tags.append([hair_color, eye_color])

        outputs = np.array(tags)
        return outputs

    def get_nextbatch(self):
        if self.images.shape[0] <= (self.batch_iter + 2) * self.batch_size:
            rand_idx = np.random.permutation(len(self.images))

            self.images = self.images[rand_idx]
            self.tags = self.tags[rand_idx]
            self.batch_iter = -1

        self.batch_iter += 1
        return self.images[self.batch_iter * self.batch_size: (self.batch_iter + 1) * self.batch_size], \
               self.tags[self.batch_iter * self.batch_size: (self.batch_iter + 1) * self.batch_size]

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


