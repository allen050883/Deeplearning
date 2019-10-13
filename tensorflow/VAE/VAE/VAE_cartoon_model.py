import tensorflow as tf
import numpy as np
import math

class MODEL:
    def __init__(self, LR, filter_num, batch_size, latent_size):
        self.LR = LR
        self.filter_num = filter_num
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.kernel = tf.keras.initializers.he_normal()

        self.x = tf.placeholder(tf.string, [None])
        
        self.dataset = tf.data.Dataset.from_tensor_slices({'imgs': self.x})
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
        self.dataset = self.dataset.prefetch(buffer_size=100)  # prefech
        self.dataset_iter = self.dataset.make_initializable_iterator()
        self.dataset_fetch = self.dataset_iter.get_next()

        def img_preprocess(x):
            img = tf.reshape( tf.image.decode_png( tf.read_file(x) )[:,:,:3], [500, 500, 3])
            img = tf.image.resize_image_with_crop_or_pad(img, 324, 324)
            img = tf.image.resize_images(img, [64, 64])
            return img
            
        self.img_full = tf.map_fn(img_preprocess, self.dataset_fetch['imgs'], dtype=tf.float32)   
        #self.img_full = ( self.img_full ) / 255
        
        self.output = self.main(self.img_full)
        self.loss_pdf = self.loss_function()
        self.loss_mse = tf.losses.mean_squared_error(self.output*255, self.img_full)
        self.loss = self.loss_mse + self.loss_pdf
        
        #self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        optimizer = tf.train.AdamOptimizer(self.LR)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        #self.prediction = tf.nn.sigmoid(self.output)
    
    def reparameterize(self, mu, logvar):
        std = tf.reshape(tf.math.exp(0.5*logvar), [self.batch_size, self.lz])
        eps = tf.random.normal(std.shape)
        return mu + eps*std
    
    def main(self, img):    
        
        img = tf.layers.conv2d(img, self.filter_num, 3, 2, 'same', kernel_initializer = self.kernel)
        img = tf.nn.tanh(tf.layers.batch_normalization(img, training = True))
        #img = tf.layers.average_pooling2d(img, 2, 2)#32, 32, 32
        
        img_1 = tf.layers.conv2d(img, self.filter_num*2, 3, 2, 'same', kernel_initializer = self.kernel)
        img_1 = tf.nn.tanh(tf.layers.batch_normalization(img_1, training = True))
        #img = tf.layers.average_pooling2d(img, 2, 2)#16, 16, 64
        
        img_2 = tf.layers.conv2d(img_1, self.filter_num*4, 3, 2, 'same', kernel_initializer = self.kernel)
        img_2 = tf.nn.tanh(tf.layers.batch_normalization(img_2, training = True))
        #img = tf.layers.average_pooling2d(img, 2, 2)#8, 8, 128
        
        img_3 = tf.layers.conv2d(img_2, self.filter_num*4, 3, 2, 'same', kernel_initializer = self.kernel)
        img_3 = tf.nn.tanh(tf.layers.batch_normalization(img_3, training = True))
        #img = tf.layers.average_pooling2d(img, 2, 2)#4, 4, 128
        
        img_shape = img_3.shape #4, 4, 128
        print(img_shape[1])
        self.img_flatten = tf.layers.flatten(img_3)
        flatten_shape = self.img_flatten.shape #4096
        
        self.lz = int(math.sqrt(self.latent_size))
        self.mean = tf.layers.dense(self.img_flatten, self.lz, 'relu')
        self.var = tf.layers.dense(self.img_flatten, self.lz, 'relu')
        self.img_flatten = self.reparameterize(self.mean, self.var)
        
        img = tf.layers.dense(self.img_flatten, flatten_shape[1])
        
        img_4 = tf.reshape(img, [-1, img_shape[1], img_shape[2], img_shape[3]])
        img_4 = tf.nn.tanh(tf.layers.batch_normalization(img_4, training = True))#4, 4, 128
        
        img_5 = tf.layers.conv2d_transpose(img_4, self.filter_num*4, 3, 2, 'same', kernel_initializer = self.kernel)
        img_5 = tf.nn.tanh(tf.layers.batch_normalization(img_5, training = True))#8, 8, 256
        
        img_6 = tf.layers.conv2d_transpose(img_5, self.filter_num*2, 3, 2, 'same', kernel_initializer = self.kernel)
        img_6 = tf.nn.tanh(tf.layers.batch_normalization(img_6, training = True))#16, 16, 256
        
        img_7 = tf.layers.conv2d_transpose(img_6, self.filter_num, 3, 2, 'same', kernel_initializer = self.kernel)
        img_7 = tf.nn.tanh(tf.layers.batch_normalization(img_7, training = True))#32, 32, 64
        
        img = tf.layers.conv2d_transpose(img_7, 3, 3, 2, 'same', kernel_initializer = self.kernel)
        img = tf.nn.sigmoid(img)
        
        return img
    
    
    def loss_function(self):
    
        def log_normal_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum(-0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
            
        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.output, labels = self.img_full)
        #logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
        #logpx_z = tf.losses.mean_squared_error(self.output*255, self.img_full)
        logpz = log_normal_pdf(self.img_flatten, 0., 0.)
        logqz_x = log_normal_pdf(self.img_flatten, self.mean, self.var)
        
        return -tf.reduce_mean(logpz - logqz_x)
