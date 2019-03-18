import tensorflow as tf

def residual_block(self, img):
        img_1 = tf.nn.relu(tf.layers.batch_normalization(img, renorm = True))
        img_1 = tf.layers.conv2d(img_1, self.filter_num, 1, 1, 'same', kernel_initializer = self.kernel)
        img_1 = tf.nn.relu(tf.layers.batch_normalization(img_1, renorm = True))
        img_1 = tf.layers.conv2d(img_1, self.filter_num, 3, 1, 'same', kernel_initializer = self.kernel)
        img_1 = tf.nn.relu(tf.layers.batch_normalization(img_1, renorm = True))
        img_1 = tf.layers.conv2d(img_1, self.filter_num*2, 1, 1, 'same', kernel_initializer = self.kernel)
            
        img_add = tf.layers.conv2d(img, self.filter_num*2, 1, 1, 'same', kernel_initializer = self.kernel)

        return img_1 + img_add
    


    def owner_block(self, img):
        img_1 = tf.layers.conv2d(img, self.filter_num, 7, 5, 'same', kernel_initializer = self.kernel)
        img_1 = tf.nn.relu(tf.layers.batch_normalization(img_1, renorm = True))

        img_2 = tf.layers.conv2d(img, self.filter_num, 5, 5, 'same', kernel_initializer = self.kernel)
        img_2 = tf.nn.relu(tf.layers.batch_normalization(img_2, renorm = True))
        
        img_3 = tf.layers.conv2d(img, self.filter_num, 1, 1, 'same', kernel_initializer = self.kernel)
        img_3 = tf.layers.max_pooling2d(img_3, 5, 5)
        img_3 = tf.nn.relu(tf.layers.batch_normalization(img_3, renorm = True))
        
        img = tf.concat([img_1, img_2, img_3], axis = -1)
        
        return img
    
        
    def ConvLayer(self, img):
        img = (img-128.)/128.
        
        img = self.owner_block(img) #(:, 400, 600, filter_num)
        img = tf.layers.conv2d(img, self.filter_num*2, 1, 1, 'same', kernel_initializer = self.kernel)
        
        img_1 = tf.layers.max_pooling2d(img, 2, 2)
        img_1 = self.residual_block(img_1) #(:, 200, 300, filter_num)
        
        img_2 = tf.layers.max_pooling2d(img_1, 2, 2)
        img_2 = self.residual_block(img_2) #(:, 100, 150, filter_num)

        img_3 = tf.layers.max_pooling2d(img_2, 2, 2)
        img_3 = self.residual_block(img_3)
        img_3 = self.residual_block(img_3)
        img_3 = self.residual_block(img_3) #(:, 50, 75, filter_num)
        img_3 = tf.image.resize_nearest_neighbor(img_3, [img_3.shape[1]*2, img_3.shape[2]*2]) #(:, 100, 150, filter_num)


        img_4 = img_2 + img_3 #(:, 100, 150, filter_num)
        img_4 = self.residual_block(img_4)
        img_4 = tf.image.resize_nearest_neighbor(img_4, [img_4.shape[1]*2, img_4.shape[2]*2]) #(:, 200, 300, filter_num)

        img_5 = img_1 + img_4 #(:, 200, 300, filter_num)
        img_5 = self.residual_block(img_5)
        img_5 = tf.image.resize_nearest_neighbor(img_5, [img_5.shape[1]*2, img_5.shape[2]*2]) #(:, 400, 600, filter_num)

        img = img + img_5 #(:, 400, 600, filter_num)


        img = self.owner_block(img) #(:, 80, 120, filter_num)
        img = tf.layers.max_pooling2d(img, (40, 40), (40, 40))
        img = tf.layers.conv2d(img, 1, 1, 1, 'same', kernel_initializer = self.kernel)
        img = tf.squeeze(img, [3])
        #print(img.shape)
        
        return img
