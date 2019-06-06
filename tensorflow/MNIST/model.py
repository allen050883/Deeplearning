import tensorflow as tf

class MODEL:
    def __init__(self, LR, filter_num, batch_size):
        self.LR = LR
        self.filter_num = filter_num
        self.batch_size = batch_size
        self.kernel = tf.keras.initializers.he_normal()
        
        with tf.variable_scope("model_input"):
            self.x = tf.placeholder(tf.float32, [None, 784], name = 'input')
            self.y = tf.placeholder(tf.float32, [None, 10])
        
        with tf.variable_scope("model"):
            self.output = self.main(self.x)
        
        self.loss = tf.losses.softmax_cross_entropy(self.y, self.output)
        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)
        
        with tf.variable_scope("result"):
            self.output = tf.nn.softmax(self.output, name = 'softmax')
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1)), tf.float32))
            self.prediction = tf.argmax(self.output, 1, name = 'prediction')
        
    
    def main(self, img):    
        img = tf.reshape(img, [-1, 28, 28, 1])
        img = tf.layers.conv2d(img, self.filter_num, 3, 2, 'same', kernel_initializer = self.kernel)
        img = tf.nn.relu(tf.layers.batch_normalization(img, training = True))
        #img = tf.layers.average_pooling2d(img, 2, 2)#32, 32, 32
        
        img = tf.layers.conv2d(img, self.filter_num*2, 3, 2, 'same', kernel_initializer = self.kernel)
        img = tf.nn.relu(tf.layers.batch_normalization(img, training = True))
        #img = tf.layers.average_pooling2d(img, 2, 2)#16, 16, 64
        
        img = tf.layers.conv2d(img, self.filter_num*4, 3, 2, 'same', kernel_initializer = self.kernel)
        img = tf.nn.relu(tf.layers.batch_normalization(img, training = True))
        #img = tf.layers.average_pooling2d(img, 2, 2)#8, 8, 128
        
        img = tf.layers.flatten(img)
        img = tf.layers.dense(img, 1024, 'relu')
        img = tf.layers.dense(img, 128, 'relu')
        img = tf.layers.dense(img, 10)
        return img
    
    