import glob
import os
import numpy as np
import tensorflow as tf

from skimage import io
from tqdm import tqdm

# Define Global params
tf.app.flags.DEFINE_integer('epochs', 256, 'Number of epochs to run')
tf.app.flags.DEFINE_integer('step_per_checkpoints', 100, 'Number of steps to save')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Size of batch')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')

tf.app.flags.DEFINE_string('model_dir', 'model/', 'Model path')
tf.app.flags.DEFINE_string('checkpoint_filename', 'vanilla_gan.ckpt', 'Checkpoint filename')


FLAGS = tf.app.flags.FLAGS

data_dir = 'faces/'

def load_data():
    files = sorted(glob.glob(data_dir + '*.jpg'))

    images = []
    for file in files:
        image = io.imread(file)
        image = np.array(image)
        images.append(image)

    # [batch, 96, 96, 3]
    images = np.array(images)
    return images

def create_batches(data):
    np.random.shuffle(data)

    batches = []
    for i in range(0, len(data), FLAGS.batch_size):
        if len(data) < i + FLAGS.batch_size:
            break
        batches.append(data[i: i + FLAGS.batch_size])

    return np.array(batches)

def main():
    # load raw image data
    images = load_data()

    with tf.Session as sess:
        # define model here
        # model = Vanilla_Gan(){
        #
        # }

        # load old model or not
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        # add summary writer for tensorboard
        summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        # training
        step_count = 0
        for epoch in range(FLAGS.epochs):

            # shuffle data and create batches
            batches = create_batches(images)

            # loop through all the batches
            for batch in tqdm(batches):
                # returns at least loss and summary
                # model.train()
                step_count += 1

                # save when the step counter % step_per_checkpoint == 0
                if step_count % FLAGS.step_per_checkpoint == 0:
                    summary_writer.add_summary(summary, step_count)
                    ckpt_file = os.path.join(FLAGS.model_dir, FLAGS.checkpoint_filename)
                    model.saver.save(sess, ckpt_file, global_step=step_count)





if __name__ == "__main__":
    main()