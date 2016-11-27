import tensorflow as tf
import numpy as np
import random
import os
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_channels', 3, "the num of channels of the input")
tf.app.flags.DEFINE_integer('input_image_size', 32, "the size of input images")
tf.app.flags.DEFINE_integer('output_target_size', 64, "the size of input images")
tf.app.flags.DEFINE_string("train_data_dir", "./dataset", "the train images dir")
tf.app.flags.DEFINE_integer("scale", 2, "the scale ratio")
tf.app.flags.DEFINE_integer("batch_size", 32, "the batch size when train data")
tf.app.flags.DEFINE_integer('max_steps', 1000, 'the num of batch to run')

def conv(input, kernel_size, output_channel, strides=[1,1,1,1], padding='VALID'):
    input_channel = input.get_shape().as_list()[3]
    kernel_shape = [kernel_size, kernel_size, input_channel, output_channel]
    kernel_weights = tf.Variable(tf.random_normal(shape=kernel_shape, stddev=1e-3), name='kernel_weights')
    conv = tf.nn.conv2d(input, kernel_weights, strides, padding=padding)
    bias_init = tf.Variable(tf.random_normal(shape=[64], name='bias'))
    conv = tf.nn.bias_add(conv, bias_init)
    conv = tf.nn.relu(conv)
    return conv, kernel_weights, bias_init

    

def srcnn_inference(images):
    """
    This script define the srcnn network structure.
    return the output images of the srcnn
    """
    # conv1
    weight_params = []
    bias_params = []
    conv1, kernel_weight1, bias_init1 = conv(images, 9, 64)
    weight_params += [kernel_weight1]
    bias_params += [bias_init1]
    conv2, kernel_weight2, bias_init2 = conv(conv1, 1, 32)
    weight_params += [kernel_weight2]
    bias_params += [bias_init2]
    conv3, kernel_weight3, bias_init3 = conv(conv2, 5, 32)
    weight_params += [kernel_weight3]
    bias_params += [bias_init3]
    return conv3, weight_params, bias_params



def get_imagesfile():
    # Return names of training files
    if not tf.gfile.Exists(FLAGS.train_data_dir) or not tf.gfile.IsDirectory(FLAGS.train_data_dir):
        print "the dir of train data is not exists"
        return

    filenames = tf.gfile.ListDirectory(FLAGS.train_data_dir)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.train_data_dir, f) for f in filenames]
    return filenames



def process_data(sess,filenames):
    """
    This script gen the input images(downsample) and labels(origin images)
    """
    images_size = FLAGS.input_image_size
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image.set_shape([None, None, channels])
    image = tf.reshape(image, [1, images_size, images_size, 3])
    image = tf.cast(image, tf.float32)/255.0
    K = FLAGS.scale
    downsampled = tf.image.resize_area(image, [images_size//K, images_size//K])
    feature = tf.reshape(downsampled, [images_size//K, images_size//K, 3])
    label = tf.reshape(image, [images_size, images_size, 3])
    features, labels = tf.train.batch([feature, label], batch_size=FLAGS.batch_size, num_threads=4, name='labels_and_features')
    tf.train.start_queue_runners(sess=sess)
    return features, labels



def inference():
    pass



def run():
    with tf.Session() as sess:
        filesname = get_imagesfile()
        features, labels = process_data(sess, filesname)
        gen_outputs, weight_params, bias_params = srcnn_inference(features)
        global_step = tf.Variable(initial_value=0)
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(labels, gen_outputs))))
        train_op = tf.train.MomentumOptimizer(0.0001,0.9).minimize(loss=loss, global_step=global_step)
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for i in range(FLAGS.max_steps):
            sess.run(train_op)
            if i %100 ==0:
                saver.save(sess, 'my-model', global_step=global_step)

        




        








