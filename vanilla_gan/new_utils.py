import numpy as np
import tensorflow as tf
import os

from scipy import misc
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord(tfrecords_filename, jpg_files):
    print("Creating TF Record")
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_file in jpg_files:
        img = np.array(Image.open(img_file))

        height = img.shape[0]
        width = img.shape[1]
        img_raw = img.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())

    writer.close()

def read_and_decode(filename_queue, FLAGS):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=FLAGS.img_size,
                                                           target_width=FLAGS.img_size)

    images = tf.train.shuffle_batch([resized_image],
                                                 batch_size=FLAGS.fixed_batch_size,
                                                 capacity=FLAGS.capacity,
                                                 num_threads=FLAGS.num_threads,
                                                 min_after_dequeue=FLAGS.min_after_dequeue)
    # Normalize
    images = tf.cast(images, tf.float32)
    images = tf.subtract(tf.truediv(images, 127.5), 1)
    return images

def immerge_save(images, epoch, img_size, path_dir):
    images = np.array(images).squeeze()
    h, w, c = images.shape[1], images.shape[2], images.shape[3]

    filename = os.path.join(path_dir, '{:d}.jpg'.format(epoch))
    img = np.zeros((h * img_size, w * img_size, c))

    for idx, image in enumerate(images):
        i = idx % img_size
        j = idx // img_size
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    img = (img + 1.) / 2

    return misc.imsave(filename, img)
