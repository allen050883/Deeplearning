import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class FPN:
    def __init__(self, LR, tf_train_mode, dp_rate, layers):
        self.tf_train_mode = tf_train_mode
        self.LR = LR
        self.dp_rate = dp_rate
        self.layers = layers

        self.ori_x = tf.placeholder(tf.string, [None], name = 'input')
        self.tf_y = tf.placeholder(tf.float32, [None, 1], name = 'label')

        self.img_full = tf.cast(tf.map_fn(lambda x: tf.reshape(tf.concat( [tf.image.decode_png( tf.read_file( x + '.png' )), tf.image.decode_png( tf.read_file( x + 's.png' )), tf.image.decode_png( tf.read_file( x + 's0.png' ))], axis = -1), [256, 256, 3]), self.ori_x, dtype=tf.uint8), tf.float32)
        
        self.output_ori = self.ConvLayer(self.img_full, self.layers)
        self.output = self.FullyConnLayer(self.output_ori)

        
        with tf.name_scope('loss'):
            self.loss = tf.losses.sigmoid_cross_entropy(self.tf_y, self.output, label_smoothing = 0.1)

        self.step = tf.Variable(0, trainable=False)
        self.rate = tf.train.exponential_decay(self.LR, self.step, 10, 0.96)
        with tf.name_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(self.rate).minimize(self.loss)

        self.prediction = tf.cast(tf.greater(tf.nn.sigmoid(self.output), 0.5), tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.tf_y), tf.float32))


    def BasicBlock(self, img, filter_num, stride):
        conv = tf.layers.conv2d(img, filter_num, 3, stride, 'same')
        bn = tf.layers.batch_normalization(conv)
        act = tf.nn.relu(bn)
        
        dp = tf.layers.dropout(act, rate = self.dp_rate, noise_shape = [ -1, tf.shape(act)[1], tf.shape(act)[1], 1 ], training = self.tf_train_mode)
        
        conv = tf.layers.conv2d(act, filter_num, 3, 1, 'same')
        bn = tf.layers.batch_normalization(conv)
        if stride!=1:
            down_img = tf.layers.conv2d(img, filter_num, 1, stride, 'same')
            down_img = tf.layers.batch_normalization(down_img)
        else:
            down_img = tf.layers.conv2d(img, filter_num, 1, stride, 'same')
            down_img = tf.layers.batch_normalization(down_img)
        
        down_img = down_img + bn
        act = tf.nn.relu(down_img)
        return act
    
    def residual(self, img, residual_name, time, filter_num, stride):
        if residual_name == 'BasicBlock':
            for t in range(time):
                if t==0:
                    img = self.BasicBlock(img, filter_num, stride)
                else:
                    img = self.BasicBlock(img, filter_num, 1)
        return img


    def ConvLayer(self, img, layers):
        img = (img-128.0)/128.0
        
        conv1 = tf.layers.conv2d(img, 64, 7, 2, 'same')#output (128, 128), 64
        bn1 = tf.layers.batch_normalization(conv1)
        act1 = tf.nn.relu(bn1)
        pool1 = tf.layers.max_pooling2d(act1, 3, 2, 'same')#output 64

        conv2_x = self.residual(pool1, 'BasicBlock', layers[0], 64, 1)#output (64, 64), 64
        conv3_x = self.residual(conv2_x, 'BasicBlock', layers[1], 128, 2)#output (32, 32), 128
        conv4_x = self.residual(conv3_x, 'BasicBlock', layers[2], 256, 2)#output (16, 16), 256
        conv5_x = self.residual(conv4_x, 'BasicBlock', layers[3], 512, 2)#output (8, 8), 512

        conv5_1x1 = tf.layers.conv2d(conv5_x, 256, 1, 1, 'same')#output (8, 8), 256
        trans5 = tf.layers.conv2d_transpose(conv5_1x1, 256, 3, 2, 'same')#output (16, 16), 256
        
        conv4_1x1 = tf.layers.conv2d(conv4_x, 256, 1, 1, 'same')#output (16, 16), 256
        trans5 = tf.layers.conv2d_transpose(conv5_1x1, 256, 3, 2, 'same')#output (16, 16), 256
        trans5 = trans5 + conv4_1x1 #output (16, 16), 256
        
        conv3_1x1 = tf.layers.conv2d(conv3_x, 128, 1, 1, 'same')#output (32, 32), 128
        trans4 = tf.layers.conv2d_transpose(trans5, 128, 3, 2, 'same')#output (32, 32), 128
        trans4 = trans4 + conv3_1x1 #output (32, 32), 128
        
        conv2_1x1 = tf.layers.conv2d(conv2_x, 64, 1, 1, 'same')#output (64, 64), 64
        trans3 = tf.layers.conv2d_transpose(trans4, 64, 3, 2, 'same')#output (64, 64), 64
        trans3 = trans3 + conv2_1x1 #output (64, 64), 64
        
        avg_5 = tf.layers.average_pooling2d(trans3, 64, 64, 'same')
        return avg_5


    def FullyConnLayer(self, ori):
        flat = tf.reshape(ori, [-1, 64])
        output = tf.layers.dense(flat, 1)
        return output
