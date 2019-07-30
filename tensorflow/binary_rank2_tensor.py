import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

path = os.getcwd() + '/20190613'
original_path = path + '/choose_ans/'

file = os.listdir(original_path)
file = [original_path + f for f in file]; file.sort()

with tf.Session() as sess: 
    img = tf.reshape( tf.image.decode_bmp( tf.read_file(file[0]) ) , (128, 128, 1))
    img_1 = tf.cast(img, tf.int32)
    img_1 = tf.mod(tf.bitwise.right_shift(tf.expand_dims(img_1, -1), tf.range(8)), 2)
    img_1 = img_1.eval()
    img = img.eval()
