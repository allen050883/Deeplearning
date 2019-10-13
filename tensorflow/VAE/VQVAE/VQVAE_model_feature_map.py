import tensorflow as tf
import numpy as np
import math

class MODEL:
    def main(self, img):
        with tf.variable_scope('encoder') as  self.enc_param_scope:
            #img (-1, 28, 28, 1)
            img_1 = tf.keras.layers.Conv2D(self.FILTER_NUM, 3, 1, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img)
            #img_1 = tf.keras.layers.BatchNormalization(renorm = True)(img_1, training = self.TRAINING_STATE)

            img_2 = tf.keras.layers.Conv2D(self.FILTER_NUM*2, 3, 1, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_1)
            #img_2 = tf.keras.layers.BatchNormalization(renorm = True)(img_2, training = self.TRAINING_STATE)

            img_3 = tf.keras.layers.Conv2D(self.FILTER_NUM*4, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_2)
            #img_3 = tf.keras.layers.BatchNormalization(renorm = True)(img_3, training = self.TRAINING_STATE)
            
            img_4 = tf.keras.layers.Conv2D(self.FILTER_NUM*4, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_3)
            #img_4 = tf.keras.layers.BatchNormalization(renorm = True)(img_4, training = self.TRAINING_STATE)
            #img_4 (-1, 7, 7, 128)
            
            z_e = tf.reshape(img_4, [-1, img_4.shape[1] * img_4.shape[2], img_4.shape[3]])
            
        with tf.variable_scope('embed') :
            vq_dictionary = tf.Variable(tf.random.uniform([self.K, img_4.shape[1].value * img_4.shape[2].value]), trainable=True, dtype=tf.float32, name='vq_dictionary')
            
            def calculate_embed(i):
                return tf.stack([vq_dictionary[tf.argmin(tf.reduce_mean(tf.pow( j - vq_dictionary, 2), axis=-1))] for j in tf.unstack(i, axis=-1)], axis=-1)
            
            z_q = tf.map_fn(calculate_embed, z_e, parallel_iterations = self.FILTER_NUM*4)
            z_q = z_e + tf.stop_gradient(z_q - z_e)
            #forward:  zq = ze + zq -ze --> zq
            #backward: zq = ze --> pass from zq to ze
            
            z_q = tf.reshape(z_q, [-1, img_4.shape[1].value, img_4.shape[2].value, img_4.shape[3].value])
            
            
        with tf.variable_scope('decoder') as self.dec_param_scope:
            img_5 = tf.keras.layers.Conv2DTranspose(self.FILTER_NUM*4, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(z_q)
            #img_5 = tf.keras.layers.BatchNormalization(renorm = True)(img_5, training = self.TRAINING_STATE)
            
            img_6 = tf.keras.layers.Conv2DTranspose(self.FILTER_NUM*2, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_5)
            #img_6 = tf.keras.layers.BatchNormalization(renorm = True)(img_6, training = self.TRAINING_STATE)
            
            img_7 = tf.keras.layers.Conv2DTranspose(self.FILTER_NUM, 3, 1, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_6)
            #img_7 = tf.keras.layers.BatchNormalization(renorm = True)(img_7, training = self.TRAINING_STATE)
            
            img_8 = tf.keras.layers.Conv2DTranspose(1, 3, 1, 'same', kernel_initializer = self.kernel)(img_7)

        return [img_8, tf.reshape(z_e, [-1, img_4.shape[1], img_4.shape[2], img_4.shape[3]]), z_q]
    
    
    def loss_function(self):
        self.recon = tf.reduce_mean(tf.pow(self.output - self.img_full, 2))  #MSE
        self.vq = tf.reduce_mean(tf.pow((tf.stop_gradient(self.VQVAE_ze) - self.VQVAE_zq), 2))
        self.commit = self.BETA * tf.reduce_mean(tf.pow(self.VQVAE_ze - tf.stop_gradient(self.VQVAE_zq), 2))
        
        #self.nll = -1.*(tf.reduce_mean(tf.log(self.img_8),axis=[1,2,3]) + tf.log(1/tf.cast(self.k, tf.float32)))/tf.log(2.)
        return self.recon + self.vq + self.commit
    
                
    def __init__(self, LR, FILTER_NUM, BETA, K, TRAINING_STATE, BATCH_SIZE):
        #LR, FILTER_NUM, BETA, K, STATE, BATCH_SIZE, LATENT_SIZE
        self.K = K
        self.LR = LR
        self.BETA = BETA
        self.FILTER_NUM = FILTER_NUM
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAINING_STATE = TRAINING_STATE
        self.kernel = tf.keras.initializers.he_uniform()

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    
        self.dataset = tf.data.Dataset.from_tensor_slices({'imgs': self.x})
        self.dataset = self.dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size = self.BATCH_SIZE))
        self.dataset = self.dataset.prefetch(buffer_size = self.BATCH_SIZE)  # prefech
        self.dataset_iter = self.dataset.make_initializable_iterator()
        self.dataset_fetch = self.dataset_iter.get_next()
        
        
        self.img_full = self.dataset_fetch['imgs']
        self.output, self.VQVAE_ze, self.VQVAE_zq = self.main(self.img_full)
        self.loss = self.loss_function()
        
        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        
       
    
    