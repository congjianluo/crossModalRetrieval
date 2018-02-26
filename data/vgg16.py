import inspect
import os

import numpy as np
import tensorflow as tf
import time
import utils
import shutil

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def searchImage(img):
    W = np.loadtxt(os.path.join('./data/txt', 'W.txt'), delimiter=',')  # r'..\data\txt\W.txt'
    img = np.dot(np.array(list(img)).astype(np.float), W)
    img = np.where(img > 0, 1, 0)

    newimg = list()

    for i in range(1, int(len(img) / 8 + 1)):
        if i == 1:
            newimg.append(int(''.join(map(str, img[7::-1])), 2))
        else:
            newimg.append(int(''.join(map(str, img[8 * i - 1:8 * (i - 1) - 1:-1])), 2))

    img = np.array(newimg).transpose()

    allbit = np.loadtxt(os.path.join('./data/txt', 'B.txt'), delimiter=',')  # r'..\data\txt\B.txt'
    allbit = allbit.astype(np.int)

    bit_in_char = '0 1 1 2 1 2 2 3 1 2 2 3 2 3 3 4 1 2 2 3 2 3\
     3 4 2 3 3 4 3 4 4 5 1 2 2 3 2 3 3 4 2 3 3 4\
     3 4 4 5 2 3 3 4 3 4 4 5 3 4 4 5 4 5 5 6 1 2\
     2 3 2 3 3 4 2 3 3 4 3 4 4 5 2 3 3 4 3 4 4 5\
     3 4 4 5 4 5 5 6 2 3 3 4 3 4 4 5 3 4 4 5 4 5\
     5 6 3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 1 2 2 3\
     2 3 3 4 2 3 3 4 3 4 4 5 2 3 3 4 3 4 4 5 3 4\
     4 5 4 5 5 6 2 3 3 4 3 4 4 5 3 4 4 5 4 5 5 6\
     3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 2 3 3 4 3 4\
     4 5 3 4 4 5 4 5 5 6 3 4 4 5 4 5 5 6 4 5 5 6\
     5 6 6 7 3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 4 5\
     5 6 5 6 6 7 5 6 6 7 6 7 7 8'

    result = []
    bit_in_char = bit_in_char.split(' ')
    for i in bit_in_char:
        if i != '':
            result.append(i)
    bit_in_char = result

    xorinfo = np.bitwise_xor(img, allbit)

    numberPic, nwords = xorinfo.shape

    hammTrainTest = np.zeros((1, numberPic))

    for i in range(0, numberPic):
        for j in range(0, nwords):
            hammTrainTest[0, i] += int(bit_in_char[xorinfo[i, j]])

    indexer = hammTrainTest.argsort().tolist()[0]
    j = 1
    for i in indexer[0:6]:
        print(i)
        shutil.copy(os.path.join('./data/allImage', str(i + 1) + '.JPEG'),
                    # r'..\data\allimage' + '\\' + str(i + 1) + r'.JPEG'
                    os.path.join('./static/img/large',
                                 str(j) + '.png'))  # r'..\static\img\large' + '\\' + str(j) + r'.png'
        shutil.copy(os.path.join('./data/allImage', str(i + 1) + '.JPEG'),
                    os.path.join('./static/img/thumb',
                                 str(
                                     j) + '.png'))  # r'..\wamp64\www\searchImage\static\img\thumb' + '\\' + str(j) + r'.png'
        j += 1


def run_vgg16(sess, filename):
    img = utils.load_image(os.path.join('./uploads', filename))
    batch = img.reshape((1, 224, 224, 3))

    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    sess = tf.Session()

    images = tf.placeholder("float", [1, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    feature = sess.run(vgg.fc7, feed_dict=feed_dict)

    sess.close()
    img = np.append(feature[0], [1])
    searchImage(img)
