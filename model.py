from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np

import scipy.io

from six.moves import xrange

from utils import read_mat
from ops import *


class Unet(object):
    def __init__(self,
                 width=256, height=256, learning_rate=0.0001,
                 data_set=None, test_set=None, result_name=None,
                 ckpt_dir=None, logs_step=None, restore_step=None,
                 hidden_num=16, epoch_num=300, batch_size=32,
                 num_gpu=1, is_train=True, w_bn=False):

        self.width = width
        self.height = height
        self.learning_rate = learning_rate

        self.data_set = data_set
        self.test_set = test_set
        if result_name is None:
            result_name = '[T]' + test_set
        self.result_name = result_name

        if ckpt_dir is None:
            ckpt_dir = data_set
        self.ckpt_dir = './logs/' + ckpt_dir
        if logs_step is None:
            self.logs_step = int(epoch_num / 5)
        else:
            self.logs_step = logs_step
        self.restore_step = restore_step

        self.hidden_num = hidden_num
        self.epoch_num = epoch_num
        self.batch_size = batch_size

        if is_train:
            self.num_gpu = num_gpu
        else:
            self.num_gpu = 1
        self.is_train = is_train
        self.w_bn = w_bn

        self.activation = relu
        self.upsampling = avg_unpool

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)

        self.writer = tf.summary.FileWriter(self.ckpt_dir)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.build_model()

    def build_model(self):
        self.X = tf.placeholder("float", [None, self.width, self.height])
        if self.is_train:
            self.trX, self.trY = read_mat(
                './data/' + self.data_set + '.mat', True)
            self.Y = tf.placeholder("float", [None, self.width, self.height])
        else:
            self.trX = read_mat('./data/' + self.test_set + '.mat', False)
        self.num_of_data = len(self.trX)

        if self.is_train:
            loss_tmp, grad_tmp = self.loss_and_grad()
            with tf.device('/cpu:0'):
                self.cost = tf.reduce_mean(loss_tmp)
                grad = average_gradients(grad_tmp)
                tf.summary.scalar("cost", self.cost)
        else:
            with tf.variable_scope(tf.get_variable_scope()):
                self.logit = self.inference(self.X)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.device('/cpu:0'):
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.apply_gradients(grad)

        self.summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        feed_dict_fixed = {self.X: self.trX[0:self.batch_size],
                           self.Y: self.trY[0:self.batch_size]}

        ground_time = time.time()
        for i in xrange(self.epoch_num):
            randpermlist = np.random.permutation(self.num_of_data)
            batch = zip(xrange(0, self.num_of_data, self.batch_size),
                        xrange(self.batch_size, self.num_of_data, self.batch_size))
            for start, end in batch:
                start_time = time.time()
                _, cost_val = self.sess.run([self.train_op, self.cost], feed_dict={
                    self.X: self.trX[randpermlist[start:end]],
                    self.Y: self.trY[randpermlist[start:end]]})
                duration = time.time() - start_time
                total_time = (time.time() - ground_time) / 60

                examples_per_sec = self.batch_size / duration

                format_str = (
                    'time: %06.2f min, epoch %03d, batch=(%04d:%04d), cost= %.2f (%.1f examples/sec)')
                print(format_str % (total_time, i, start,
                                    end, cost_val, examples_per_sec))

            summary_str = self.sess.run(
                self.summary, feed_dict=feed_dict_fixed)
            self.writer.add_summary(summary_str, global_step=i + 1)

            if (i + 1) % self.logs_step == 0:
                self.saver.save(self.sess, self.ckpt_dir +
                                '/model.ckpt', global_step=i + 1)

    def test(self):
        if self.restore_step is None:
            self.saver.restore(
                self.sess, tf.train.latest_checkpoint(self.ckpt_dir + '/'))
        else:
            self.saver.restore(self.sess, self.ckpt_dir +
                               '/model.ckpt-' + str(self.restore_step))

        batch = zip(xrange(0, self.num_of_data, self.batch_size),
                    xrange(self.batch_size, self.num_of_data, self.batch_size)
                    + [self.num_of_data])
        out_ = []
        for start, end in batch:
            tmp_out = self.sess.run(self.logit, feed_dict={
                                    self.X: self.trX[start:end]})
            out_.append(tmp_out)
        Out_ = self.sess.run(tf.concat(axis=0, values=out_))

        scipy.io.savemat(self.ckpt_dir + '/' + self.result_name +
                         '-' + str(self.restore_step) + '.mat', {'output': Out_})
        print(Out_.shape)

    def loss_and_grad(self):
        # Calculate the gradients and losses for each model tower.
        loss_tmp = []
        grad_tmp = []
        mb_per_gpu = int(self.batch_size / self.num_gpu)
        with tf.variable_scope(tf.get_variable_scope()):
            for d in range(self.num_gpu):
                with tf.device('/gpu:' + str(d)):
                    # Calculate the loss for one tower of the model. This function
                    # constructs the entire model but shares the variables across all towers.
                    mb_start = mb_per_gpu * d
                    mb_end = mb_per_gpu * (d + 1)
                    label_mb = self.Y[mb_start:mb_end]
                    image_mb = self.inference(input_=self.X[mb_start:mb_end])
                    loss = tf.nn.l2_loss(label_mb - image_mb) / mb_per_gpu

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the batch of data on this tower.
                    g_v=self.optimizer.compute_gradients(loss)

                    # Keep track of the gradients and loss across all towers.
                    loss_tmp.append(loss)
                    grad_tmp.append(g_v)
        return loss_tmp, grad_tmp

    def inference(self, input_):
        h_=self.hidden_num
        input_=tf.reshape(input_, [-1, self.width, self.height, 1])

        conv1, pool1=conv_conv_pool(input_, [h_, h_], self.activation,
                                        self.w_bn, self.is_train, name = "1")

        conv2, pool2=conv_conv_pool(pool1, [2 * h_, 2 * h_], self.activation,
                                        self.w_bn, self.is_train,  name = "2")

        conv3=conv_conv_pool(pool2, [4 * h_, 2 * h_], self.activation,
                                self.w_bn, self.is_train, name="3", pool=False)

        up4 = tf.concat([self.upsampling(conv3), conv2], 3)
        conv4 = conv_conv_pool(up4, [4 * h_, 2 * h_], self.activation,
                                self.w_bn, self.is_train, name="4", pool=False,)

        up5 = tf.concat([self.upsampling(conv4), conv1], 3)
        conv5 = conv_conv_pool(up5, [2 * h_, h_], self.activation,
                                self.w_bn, self.is_train, name="5", pool=False)

        conv6 = conv2d(conv5, 1, k_h=1, k_w=1, name='layer_fc')

        return tf.reshape(conv6, [-1, self.width, self.height])
