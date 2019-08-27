import tensorflow as tf
import numpy as np
import math

class MODEL:
    def main(self, img):
        with tf.variable_scope('encoder') as  self.enc_param_scope:
            #img (-1, 28, 28, 1)
            img_1 = tf.keras.layers.Conv2D(self.FILTER_NUM, 3, 1, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img)
            img_1 = tf.keras.layers.BatchNormalization(renorm = True)(img_1, training = self.TRAINING_STATE)

            img_2 = tf.keras.layers.Conv2D(self.FILTER_NUM*2, 3, 1, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_1)
            img_2 = tf.keras.layers.BatchNormalization(renorm = True)(img_2, training = self.TRAINING_STATE)

            img_3 = tf.keras.layers.Conv2D(self.FILTER_NUM*4, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_2)
            img_3 = tf.keras.layers.BatchNormalization(renorm = True)(img_3, training = self.TRAINING_STATE)
            
            img_4 = tf.keras.layers.Conv2D(self.FILTER_NUM*4, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_3)
            img_4 = tf.keras.layers.BatchNormalization(renorm = True)(img_4, training = self.TRAINING_STATE)
            #img_4 (-1, 7, 7, 128)
            
            z_e_bottom = tf.reshape(img_4, [-1, img_4.shape[1] * img_4.shape[2], img_4.shape[3]])
            
            
        with tf.variable_scope('vq_bottom') :
            vq_dictionary_bottom = tf.Variable(tf.random.uniform([self.K_bottom, img_4.shape[3].value]), trainable=True, dtype=tf.float32, name='vq_dictionary_bottom')
            
            def calculate_embed_bottom(i):
                return tf.stack([vq_dictionary_bottom[tf.argmin(tf.reduce_mean(tf.pow( j - vq_dictionary_bottom, 2), axis=-1))] for j in tf.unstack(i, axis = 0)], axis = 0)
            
            z_q_bottom = tf.map_fn(calculate_embed_bottom, z_e_bottom, parallel_iterations = self.FILTER_NUM*4)
            z_q_bottom = z_e_bottom + tf.stop_gradient(z_q_bottom - z_e_bottom)
            #forward:  zq = ze + zq -ze --> zq
            #backward: zq = ze --> pass from zq to ze
            
            z_q_bottom = tf.reshape(z_q_bottom, [-1, img_4.shape[1].value, img_4.shape[2].value, img_4.shape[3].value])
        
        
        with tf.variable_scope('decoder_vq_bottom') as self.dec_param_scope:
            img_5 = tf.keras.layers.Conv2DTranspose(self.FILTER_NUM*4, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(z_q_bottom)
            img_5 = tf.keras.layers.BatchNormalization(renorm = True)(img_5, training = self.TRAINING_STATE)
            
            img_6 = tf.keras.layers.Conv2DTranspose(self.FILTER_NUM*2, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(img_5)
            img_6 = tf.keras.layers.BatchNormalization(renorm = True)(img_6, training = self.TRAINING_STATE)
            
        with tf.variable_scope('vq_top') :
            img_6 = tf.concat([img_2, img_6], axis = -1)
            z_e_top = tf.reshape(img_6, [-1, img_6.shape[1] * img_6.shape[2], img_6.shape[3]])
            
            vq_dictionary_top = tf.Variable(tf.random.uniform([self.K_top, img_6.shape[3].value]), trainable=True, dtype=tf.float32, name='vq_dictionary_top')
            
            def calculate_embed_top(i):
                return tf.stack([vq_dictionary_top[tf.argmin(tf.reduce_mean(tf.pow( j - vq_dictionary_top, 2), axis=-1))] for j in tf.unstack(i, axis = 0)], axis = 0)
            z_q_top = tf.map_fn(calculate_embed_top, z_e_top, parallel_iterations = img_6.shape[3].value)
            z_q_top = z_e_top + tf.stop_gradient(z_q_top - z_e_top)
            #forward:  zq = ze + zq -ze --> zq
            #backward: zq = ze --> pass from zq to ze
            
            z_q_top = tf.reshape(z_q_top, [-1, img_6.shape[1].value, img_6.shape[2].value, img_6.shape[3].value])

        
        with tf.variable_scope('decoder_bottom') as self.dec_param_scope:
            z_q_bottom_up = tf.keras.layers.Conv2DTranspose(self.FILTER_NUM*4, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(z_q_bottom)
            z_q_bottom_up = tf.keras.layers.BatchNormalization(renorm = True)(z_q_bottom_up, training = self.TRAINING_STATE)
            z_q_bottom_up = tf.keras.layers.Conv2DTranspose(self.FILTER_NUM*2, 3, 2, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(z_q_bottom_up)
            z_q_bottom_up = tf.keras.layers.BatchNormalization(renorm = True)(z_q_bottom_up, training = self.TRAINING_STATE)
            
            
            z_q = tf.concat([z_q_bottom_up, z_q_top], axis = -1)
            
            img_7 = tf.keras.layers.Conv2DTranspose(self.FILTER_NUM, 3, 1, 'same', kernel_initializer = self.kernel, activation = tf.nn.relu)(z_q)
            img_7 = tf.keras.layers.BatchNormalization(renorm = True)(img_7, training = self.TRAINING_STATE)
            
            img_8 = tf.keras.layers.Conv2DTranspose(1, 3, 1, 'same', activation = tf.nn.sigmoid)(img_7)

        return [img_8, tf.reshape(z_e_bottom, [-1, img_4.shape[1], img_4.shape[2], img_4.shape[3]]), z_q_bottom, tf.reshape(z_e_top, [-1, img_6.shape[1], img_6.shape[2], img_6.shape[3]]), z_q_top]
    
    
    def loss_function(self):
        self.recon = tf.reduce_mean(tf.pow(self.output - self.img_full, 2))  #MSE
        self.vq_bottom = tf.reduce_mean(tf.pow((tf.stop_gradient(self.VQVAE_ze_bottom) - self.VQVAE_zq_bottom), 2))
        self.commit_bottom = self.BETA * tf.reduce_mean(tf.pow(self.VQVAE_ze_bottom - tf.stop_gradient(self.VQVAE_zq_bottom), 2))
        self.vq_top = tf.reduce_mean(tf.pow((tf.stop_gradient(self.VQVAE_ze_top) - self.VQVAE_zq_top), 2))
        self.commit_top = self.BETA * tf.reduce_mean(tf.pow(self.VQVAE_ze_top - tf.stop_gradient(self.VQVAE_zq_top), 2))
        return self.recon + self.vq_bottom + self.commit_bottom + self.vq_top + self.commit_top
    
                
    def __init__(self, LR, FILTER_NUM, BETA, K_bottom, K_top, TRAINING_STATE, BATCH_SIZE):
        #LR, FILTER_NUM, BETA, K, STATE, BATCH_SIZE, LATENT_SIZE
        self.K_bottom = K_bottom
        self.K_top = K_top
        self.LR = LR
        self.BETA = BETA
        self.FILTER_NUM = FILTER_NUM
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAINING_STATE = TRAINING_STATE
        self.kernel = tf.keras.initializers.he_normal()

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    
        self.dataset = tf.data.Dataset.from_tensor_slices({'imgs': self.x})
        self.dataset = self.dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        #self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size = self.BATCH_SIZE))
        self.dataset = self.dataset.prefetch(buffer_size = self.BATCH_SIZE)  # prefech
        self.dataset_iter = self.dataset.make_initializable_iterator()
        self.dataset_fetch = self.dataset_iter.get_next()
        
        
        self.img_full = self.dataset_fetch['imgs']
        self.output, self.VQVAE_ze_bottom, self.VQVAE_zq_bottom, self.VQVAE_ze_top, self.VQVAE_zq_top = self.main(self.img_full)
        self.loss = self.loss_function()
        
        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        
       
    
    
