#read data and transfer to npy
import os
import numpy as np
import tensorflow as tf
from model import MODEL
from tensorflow.python.framework import graph_util

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

#hyperparameter
EPOCH = 1
BATCH_SIZE = 250
FILTER_NUM = 32
LR = 1e-3

from keras.utils.np_utils import to_categorical   
y_train = to_categorical(y_train, num_classes=10)

# create session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


class MODEL:
    def __init__(self, LR, filter_num, batch_size):
        self.LR = LR
        self.filter_num = filter_num
        self.batch_size = batch_size
        self.kernel = tf.keras.initializers.he_uniform()
        
        self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name = 'inputs')
        self.labels = tf.placeholder(tf.float32, [None, 10], name = 'labels')
        
        with tf.variable_scope("model"):
            self.output = self.main(self.inputs)
            self.output = tf.identity(self.output, name = 'output')
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels, logits = self.output), name = 'loss')
            self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        
        with tf.variable_scope("result"):
            self.softmax = tf.nn.softmax(self.output, name = 'softmax')
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.softmax,1), tf.argmax(self.labels, 1)), tf.float32))
            self.prediction = tf.argmax(self.softmax, 1, name = 'prediction')
        
    
    def main(self, img):    
        img = tf.keras.layers.Conv2D(self.filter_num, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img)
        img = tf.keras.layers.Conv2D(self.filter_num*2, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img)
        img = tf.keras.layers.Conv2D(self.filter_num*4, 3, 1, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img)
        
        img = tf.keras.layers.Flatten()(img)
        img = tf.keras.layers.Dense(1024, 'relu')(img)
        img = tf.keras.layers.Dense(128, 'relu')(img)
        img = tf.keras.layers.Dense(10)(img)
        return img


tf.reset_default_graph()
model = MODEL(LR, FILTER_NUM, BATCH_SIZE)
saver = tf.train.Saver(max_to_keep = 1)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    
    #training
    for e in range(EPOCH):
        acc_list = []; loss_list = []
        for step in range( int( len(x_train) / BATCH_SIZE ) ):
            x = x_train[ BATCH_SIZE * step : BATCH_SIZE * (step+1) ]
            y = y_train[ BATCH_SIZE * step : BATCH_SIZE * (step+1) ]
            _, acc, loss = sess.run([model.train_op, model.accuracy, model.loss], feed_dict={model.inputs: x, model.labels: y})
            acc_list.append(acc); loss_list.append(loss)
        print('epoch: ', e+1, " loss: ", np.mean(loss_list), " acc: ", np.mean(acc_list))
        

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["result/prediction"])
        with tf.gfile.FastGFile("mnist.pb", mode='wb') as f:
            f.write(constant_graph.SerializeToString())


graph_def_file = "mnist.pb"
input_arrays = ["inputs"]
output_arrays = ["result/prediction"]
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("mnist.tflite", "wb").write(tflite_model)
          
    
