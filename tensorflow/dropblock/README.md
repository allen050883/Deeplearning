#The methods to use dropblock and result
import os
import cv2
import tensorflow as tf
from dropblock import DropBlock2D
from dropblock_white import DropBlock2D_white
from dropblock_gray import DropBlock2D_gray

path = os.getcwd() + '/area_patch_test_equalizehist/'
img = tf.reshape( tf.image.decode_bmp( tf.read_file(path + '25_d_14.bmp') ) , (1, 128, 128, 1))
img = tf.cast(img, tf.float32)

drop_block = DropBlock2D(keep_prob = 0.8, block_size=3)
img1 = drop_block(img, True)
img1 = tf.reshape(tf.cast(img1, tf.uint8), (128, 128, 1))

drop_block_white = DropBlock2D_white(keep_prob = 0.8, block_size=3)
img2 = drop_block_white(img, True)
img2 = tf.reshape(tf.cast(img2, tf.uint8), (128, 128, 1))

img3 = (img-127.5) / 127.5
drop_block_gray = DropBlock2D_gray(keep_prob = 0.8, block_size=3)
img3 = drop_block_gray(img3, True)
img3 = img3 * 127.5 + 127.5
img3 = tf.reshape(tf.cast(img3, tf.uint8), (128, 128, 1))

with tf.Session() as sess:
    im1 = sess.run(img1)
    im2 = sess.run(img2)
    im3 = sess.run(img3)

cv2.imwrite('im1.jpg', im1)
cv2.imwrite('im2.jpg', im2)
cv2.imwrite('im3.jpg', im3)
