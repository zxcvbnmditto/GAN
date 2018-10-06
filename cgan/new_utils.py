import numpy as np
import pandas as pd
import tensorflow as tf
import os

from scipy import misc
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord(tfrecords_filename, jpg_files, tags_file):
    print("Creating TF Record")
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    with open(tags_file, 'r') as f:
        reader = pd.read_csv(f, skiprows=[])
        reader = np.array(reader)
        for img_file, tags in zip(jpg_files, reader):
            img = np.array(Image.open(img_file))
            feature = np.array(tags[1:])
            feature = feature.astype(int)

            feature_length = len(feature)
            height = img.shape[0]
            width = img.shape[1]
            img_raw = img.tostring()
            feature = feature.tostring()

            print(feature_length, feature)

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'feature_length': _int64_feature(feature_length),
                'feature_raw': _bytes_feature(feature),
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
            'feature_length': tf.FixedLenFeature([], tf.int64),
            'feature_raw': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    feature_length = tf.cast(features['feature_length'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    feature = tf.decode_raw(features['feature_raw'], tf.int64)


    feature_shape = tf.stack([22])
    feature = tf.reshape(feature, feature_shape)

    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=FLAGS.img_size,
                                                           target_width=FLAGS.img_size)

    images, features = tf.train.shuffle_batch([resized_image, feature],
                                                 batch_size=FLAGS.fixed_batch_size,
                                                 capacity=FLAGS.capacity,
                                                 num_threads=FLAGS.num_threads,
                                                 min_after_dequeue=FLAGS.min_after_dequeue)

    # Normalize
    images = tf.cast(images, tf.float32)
    images = tf.subtract(tf.truediv(images, 127.5), 1)
    return images, features

def add_sample_noise(inputs):
    return inputs + 0.5 * inputs.std() * np.random.random(inputs.shape)

def sample_tags(hair_dict, eye_dict):
    tags = []
    for i in range(len(hair_dict)):
        for j in range(len(eye_dict)):
            tag = np.zeros(len(hair_dict) + len(eye_dict))
            tag[i] = 1
            tag[len(hair_dict) + j] = 1
            tags.append(tag)

    return np.array(tags)

def immerge_save(images, epoch, width, height, path_dir):
    images = np.array(images).squeeze()
    h, w, c = images.shape[1], images.shape[2], images.shape[3]

    filename = os.path.join(path_dir, '{:d}.jpg'.format(epoch))
    img = np.zeros((h * height, w * width, c))

    for idx, image in enumerate(images):
        i = idx % width
        j = idx // width
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    img = (img + 1.) / 2

    return misc.imsave(filename, img)
