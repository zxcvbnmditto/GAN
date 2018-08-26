import numpy as np
import glob
import os

from skimage import io
from scipy import misc

data_dirs = ['../../faces/64-64/', '../../faces/extra/']

class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.batch_iter = -1
        self.c_dim = 3
        self.images = self.load_data()

    def load_data(self):
        print("LOADING IMAGES ........")
        images = []
        for data_dir in data_dirs:
            files = sorted(glob.glob(data_dir + '*.jpg'))
            for file in files:
                image = io.imread(file)
                image = np.array(image)
                images.append(image)

        # [batch, 64, 64, 3]
        images = np.array(images)
        print(images.shape)
        outputs = (images / 127.5) - 1
        return outputs

    def get_nextbatch(self):
        if self.images.shape[0] <= (self.batch_iter + 2) * self.batch_size:
            np.random.shuffle(self.images)
            self.batch_iter = -1

        self.batch_iter += 1
        return self.images[self.batch_iter * self.batch_size: (self.batch_iter + 1) * self.batch_size]

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