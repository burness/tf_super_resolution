import tensorflow as tf
import numpy as np
import random
import os
import time
from PIL import Image as im
import logging
head = '%(asctime)-15s (message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer(
    'image_channels', 3, "the num of channels of the input")
tf.app.flags.DEFINE_integer('input_image_size', 224,
                            "the size of input images")
tf.app.flags.DEFINE_integer(
    'output_target_size', 224, "the size of input images")
tf.app.flags.DEFINE_string(
    "train_data_dir", "./dataset", "the train images dir")
tf.app.flags.DEFINE_integer("scale", 2, "the scale ratio")
tf.app.flags.DEFINE_integer("batch_size", 32, "the batch size when train data")
tf.app.flags.DEFINE_integer('max_steps', 1000, 'the num of batch to run')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', ".", "the checkpoint dir to restore")
tf.app.flags.DEFINE_string('mode', 'train', 'the mode of the script')
tf.app.flags.DEFINE_integer('save_steps', 10000, "the num of steps to save")


def conv_factory(input, kernel_size, output_channel, strides=[1, 1, 1, 1], padding='SAME', is_relu=True):
    """
    the conv function
    """
    input_channel = input.get_shape().as_list()[3]
    kernel_shape = [kernel_size, kernel_size, input_channel, output_channel]
    kernel_weights = tf.Variable(tf.random_normal(
        shape=kernel_shape, stddev=1e-3), name='kernel_weights')
    conv = tf.nn.conv2d(input, kernel_weights, strides, padding=padding)
    bias_init = tf.Variable(tf.zeros(
        shape=[output_channel], name='bias'))
    conv = tf.nn.bias_add(conv, bias_init)
    if is_relu:
        conv = tf.nn.relu(conv)
    return conv, kernel_weights, bias_init


def srcnn(images):
    """
    This script define the srcnn network structure.
    return the output images of the srcnn
    """
    # conv1
    weight_params = []
    bias_params = []
    conv1, kernel_weight1, bias_init1 = conv_factory(images, 9, 64)
    print conv1.get_shape()
    weight_params += [kernel_weight1]
    bias_params += [bias_init1]
    conv2, kernel_weight2, bias_init2 = conv_factory(conv1, 1, 32)
    print conv2.get_shape()
    weight_params += [kernel_weight2]
    bias_params += [bias_init2]
    conv3, kernel_weight3, bias_init3 = conv_factory(
        conv2, 5, 3, is_relu=False)
    weight_params += [kernel_weight3]
    bias_params += [bias_init3]
    return conv3, weight_params, bias_params


def get_imagesfile():
    """
    Return names of training files for `tf.train.string_input_producer`
    """
    if not tf.gfile.Exists(FLAGS.train_data_dir) or not tf.gfile.IsDirectory(FLAGS.train_data_dir):
        print "the dir of train data is not exists"
        return

    filenames = tf.gfile.ListDirectory(FLAGS.train_data_dir)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.train_data_dir, f) for f in filenames]
    return filenames


def process_data(sess, filenames):
    """
    This script gen the input images(downsample) and labels(origin images)
    """
    images_size = FLAGS.input_image_size
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    _, value = reader.read(filename_queue)
    channels = FLAGS.image_channels
    image = tf.image.decode_jpeg(
        value, channels=channels, name="dataset_image")
    # add data augmentation here
    image.set_shape([None, None, channels])
    image = tf.reshape(image, [1, images_size, images_size, 3])
    image = tf.cast(image, tf.float32) / 255.0
    K = FLAGS.scale
    downsampled = tf.image.resize_area(
        image, [images_size // K, images_size // K])
    upsampled = tf.image.resize_area(downsampled, [images_size, images_size])

    feature = tf.reshape(upsampled, [images_size, images_size, 3])
    label = tf.reshape(image, [images_size, images_size, 3])
    features, labels = tf.train.shuffle_batch(
        [feature, label], batch_size=FLAGS.batch_size, num_threads=4, capacity=5000, min_after_dequeue=1000, name='labels_and_features')
    tf.train.start_queue_runners(sess=sess)
    print 'tag31', features.eval(), labels.get_shape()
    return features, labels


def inference(image_file):
    """
    image_file : `String` the name of the inference file, now support the jpg files
    """
    if image_file.endswith(".bmp") or image_file.endswith(".jpg") or image_file.endswith(".png"):
        image = im.open(image_file)
        # print np.array(image)
        K = FLAGS.scale
        width, height = image.size
        image_downsampling = image.resize(
            [width // K, height // K], im.BICUBIC)
        image_downsampling.save('out_ds.jpg')
        image_upsampling = image_downsampling.resize(
            [width, height], im.BICUBIC)
        image_upsampling.save('out_us.jpg')
        image_array = np.asarray(image_upsampling)
        print 'tag', image_array.shape
        image_array = np.expand_dims(image_array[:, :, :], axis=0)
        print 'tag2', image_array.shape
        print 'tag3', image_array
        # downsampling and upsampling
    else:
        print "bad test image file"
        exit()
    images = tf.placeholder(tf.float32, shape=(
        1, image_array.shape[1], image_array.shape[2], FLAGS.image_channels))
    conv, _, _ = srcnn(images)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        else:
            print "no checkpoint in checkpoint dir"
        out_image = sess.run([conv], feed_dict={images: image_array})
        print out_image
        # out_image = out_image[0][0, :, :, 0]
        out_image = out_image[0][0, :, :, :]
        out_image = im.fromarray(out_image, 'RGB')
        # im2 = Image.new('RGB', out_image.shape)
        # im2.putdata(list_of_pixels)
        out_image.save('out.jpg')


def run():
    """
    The function to run the train
    """
    with tf.Session() as sess:
        filesname = get_imagesfile()
        print filesname
        features, labels = process_data(sess, filesname)
        gen_outputs, _, _ = srcnn(features)
        global_step = tf.Variable(initial_value=0)
        print labels.get_shape(), gen_outputs.get_shape()
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(labels, gen_outputs))))
        train_op = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(
            loss=loss, global_step=global_step)
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for i in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_val = sess.run([train_op, loss])
            print "the step {0} takes time {1}".format(i, time.time() - start_time)
            print "the step {0} loss: {1}".format(i, loss_val)
            if i % FLAGS.save_steps == 0:
                print "saving model {0}".format(i)
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir,
                                              'my-model'), global_step=global_step)


def main(argv):
    if 'train' == FLAGS.mode:
        run()
    elif 'inference' == FLAGS.mode:
        inference('dataset/102flowers/image_04083.jpg')

if __name__ == "__main__":
    tf.app.run()
