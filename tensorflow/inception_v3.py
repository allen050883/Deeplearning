import tensorflow as tf

kernel = tf.contrib.layers.xavier_initializer()
filter_num = 32

def Inception_1(self, img):
    def module_1(img):
        conv1 = tf.layers.conv2d(img, filter_num*2, 1, 1, 'same', kernel_initializer = kernel)
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
   
        conv2 = tf.layers.average_pooling2d(img, 3, 1, "same")
        conv2 = tf.layers.conv2d(conv2, filter_num, 1, 1, 'same', kernel_initializer = kernel)
        conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2))
        
        conv3 = tf.layers.conv2d(img, int(filter_num*1.5), 1, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*2, 3, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))

        conv4 = tf.layers.conv2d(img, filter_num*2, 1, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*3, 3, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*3, 3, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        
        concat = tf.concat([conv1, conv2, conv3, conv4], -1)
        return concat
        
    
    def module_2(img):
        conv1 = tf.layers.conv2d(img, filter_num*2, 1, 1, 'same', kernel_initializer = kernel)
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
        
        conv2 = tf.layers.average_pooling2d(img, 3, 1, "same")
        conv2 = tf.layers.conv2d(conv2, filter_num*2, 1, 1, 'same', kernel_initializer = kernel)
        conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2))
        
        conv3 = tf.layers.conv2d(img, int(filter_num*1.5), 1, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*2, 3, 1, 'same', kernel_initializer = kernel)

        conv4 = tf.layers.conv2d(img, filter_num*2, 1, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*3, 3, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*3, 3, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        
        concat = tf.concat([conv1, conv2, conv3, conv4], -1)
        return concat
    
    m1 = module_1(img)
    m2 = module_2(m1)
    m3 = module_2(m2)
    return m3


def Inception_2(self, img):
    def module_1(img):
        conv1 = tf.layers.conv2d(img, filter_num*12, 3, 2, 'same', kernel_initializer = kernel)
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
        
        conv2 = tf.layers.average_pooling2d(img, 3, 2, "same")
        
        conv3 = tf.layers.conv2d(img, filter_num*2, 1, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*3, 3, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*3, 3, 2, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        
        concat = tf.concat([conv1, conv2, conv3], -1)
        return concat
        
    
    def module_2(img):
        conv1 = tf.layers.conv2d(img, filter_num*6, 1, 1, 'same', kernel_initializer = kernel)
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
        
        conv2 = tf.layers.average_pooling2d(img, 3, 1, "same")
        conv2 = tf.layers.conv2d(conv2, filter_num*6, 1, 1, 'same', kernel_initializer = kernel)
        conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2))
        
        conv3 = tf.layers.conv2d(img, filter_num*4, 1, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*4, [1, 7], 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*6, [7, 1], 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
                    
        conv4 = tf.layers.conv2d(img, filter_num*4, 1, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*4, [1, 7], 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*4, [7, 1], 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*4, [1, 7], 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*6, [7, 1], 1, 'same', kernel_initializer = kernel)
        
        concat = tf.concat([conv1, conv2, conv3, conv4], -1)
        return concat

        
    def module_3(img):
        conv1 = tf.layers.conv2d(img, filter_num*6, 1, 1, 'same', kernel_initializer = kernel)
        
        conv2 = tf.layers.average_pooling2d(img, 3, 1, "same")
        conv2 = tf.layers.conv2d(conv2, filter_num*6, 1, 1, 'same', kernel_initializer = kernel)
        conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2))
        
        conv3 = tf.layers.conv2d(img, filter_num*5, 1, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*5, [1, 7], 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*6, [7, 1], 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        
        conv4 = tf.layers.conv2d(img, filter_num*5, 1, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*5, [1, 7], 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*5, [7, 1], 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*5, [1, 7], 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*6, [7, 1], 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        
        concat = tf.concat([conv1, conv2, conv3, conv4], -1)

        return concat
    
    
    m1 = module_1(img)
    m2 = module_2(m1)
    m3 = module_3(m2)
    m4 = module_3(m3)
    m5 = module_3(m4)
    
    return m5


def Inception_3(self, img):
    def module_1(img):
        conv1 = tf.layers.conv2d(img, filter_num*6, 1, 1, 'same', kernel_initializer = kernel)
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
        conv1 = tf.layers.conv2d(conv1, filter_num*10, 3, 2, 'same', kernel_initializer = kernel)
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
        
        conv2 = tf.layers.max_pooling2d(img, 3, 2, "same")
        
        conv3 = tf.layers.conv2d(img, filter_num*6, 1, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*6, [1, 7], 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*6, [7, 1], 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3 = tf.layers.conv2d(conv3, filter_num*6, 3, 2, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))

        concat = tf.concat([conv1, conv2, conv3], -1)
        
        return concat
        
    
    def module_2(img):
        conv1 = tf.layers.conv2d(img, filter_num*10, 1, 1, 'same', kernel_initializer = kernel)
        conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1))
        
        conv2 = tf.layers.average_pooling2d(img, 3, 1, "same")
        conv2 = tf.layers.conv2d(conv2, filter_num*6, 1, 1, 'same', kernel_initializer = kernel)
        conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2))
        
        conv3 = tf.layers.conv2d(img, filter_num*12, 1, 1, 'same', kernel_initializer = kernel)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(conv3))
        conv3_2 = tf.layers.conv2d(conv3, filter_num*12, [1, 3], 1, 'same', kernel_initializer = kernel)
        conv3_2 = tf.nn.relu(tf.layers.batch_normalization(conv3_2))
        conv3_3 = tf.layers.conv2d(conv3, filter_num*12, [3, 1], 1, 'same', kernel_initializer = kernel)
        conv3_3 = tf.nn.relu(tf.layers.batch_normalization(conv3_3))
        conv3 = tf.concat([conv3_2, conv3_3], -1)
        
        conv4 = tf.layers.conv2d(img, filter_num*14, 1, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4 = tf.layers.conv2d(conv4, filter_num*12, 3, 1, 'same', kernel_initializer = kernel)
        conv4 = tf.nn.relu(tf.layers.batch_normalization(conv4))
        conv4_3 = tf.layers.conv2d(conv4, filter_num*12, [1, 3], 1, 'same', kernel_initializer = kernel)
        conv4_3 = tf.nn.relu(tf.layers.batch_normalization(conv4_3))
        conv4_4 = tf.layers.conv2d(conv4, filter_num*12, [3, 1], 1, 'same', kernel_initializer = kernel)
        conv4_4 = tf.nn.relu(tf.layers.batch_normalization(conv4_4))
        conv4 = tf.concat([conv4_3, conv4_4], -1)
        
        concat = tf.concat([conv1, conv2, conv3, conv4], -1)
        
        return concat
    
    
    m1 = module_1(img)
    m2 = module_2(m1)
    m3 = module_2(m2)
    
    return m3
    

img = tf.zeros([1, 299, 299, 3], tf.float32)
img = tf.layers.conv2d(img, 32, 3, 2, 'valid', kernel_initializer = kernel)
img = tf.layers.conv2d(img, 32, 3, 1, 'same', kernel_initializer = kernel)
img = tf.layers.conv2d(img, 64, 3, 1, 'same', kernel_initializer = kernel)
img = tf.layers.max_pooling2d(img, 3, 2, 'same')
img = tf.layers.conv2d(img, 80, 3, 2, 'valid', kernel_initializer = kernel)
img = tf.layers.conv2d(img, 192, 3, 2, 'same', kernel_initializer = kernel)
img = tf.layers.conv2d(img, 288, 3, 1, 'same', kernel_initializer = kernel)

img = Inception_1(img)
img = Inception_2(img)
img = Inception_3(img)

img = tf.layers.max_pooling2d(img, 8, 8, 'same')
img = tf.layers.conv2d(img, 1000, 1, 1, 'same', kernel_initializer = kernel)
