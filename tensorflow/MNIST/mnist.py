#read data and transfer to npy
import os
import warnings
import numpy as np
import tensorflow as tf
from model import MODEL
from skimage import io
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
X_train = mnist.train.images
Y_train = mnist.train.labels

#hyperparameter
EPOCH = 1
BATCH_SIZE = 100
FILTER_NUM = 32
LR = 1e-4

warnings.filterwarnings("ignore")
print("This is EPOCH :", EPOCH)

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

#new data
if os.path.exists("./new_data/"):
    file_all = os.listdir("./new_data/")
    for f_a in file_all:
        file = os.listdir("./new_data/"+f_a+"/")
        print(f_a+" has "+str(len(file))+" files")
        for f in file:
            new = io.imread("./new_data/"+f_a+"/"+f)
            new = new.astype(np.float32)
            if len(new.shape) > 2:
                new = new[:,:,0]
            new = np.reshape(new, (1, 28*28))
            X_train = np.concatenate([X_train, new], axis = 0) 
            Y_train = np.concatenate([Y_train, convert_to_one_hot(np.array([int(f_a)]), len(file_all))])
    print("X_train and Y_train concatenate finish!")
else:
    print("There is no new_data file.")
#%%

# create session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


tf.reset_default_graph()
model = MODEL(LR, FILTER_NUM, BATCH_SIZE)
saver = tf.train.Saver(max_to_keep = 1)
with tf.Session(config=config) as sess:
    #saver.restore(sess, './AOI/model/'+'0308'+'/tf_AOI_n2n_'+'1'+'.ckpt')
    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("TensorBoard/", sess.graph)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    #training
    print("Total steps: " + str( int( mnist.train.images.shape[0] / BATCH_SIZE ) ) )
    for e in range(EPOCH):
        
        rand = np.random.randint(1000)
        np.random.seed(rand); np.random.shuffle(X_train)
        np.random.seed(rand); np.random.shuffle(Y_train)
        for step in range( int( mnist.train.images.shape[0] / BATCH_SIZE ) ):
            train = X_train[step * BATCH_SIZE :( step * BATCH_SIZE + BATCH_SIZE )]
            label = Y_train[step * BATCH_SIZE :( step * BATCH_SIZE + BATCH_SIZE )]
			
            _, acc, loss = sess.run([model.train_op, model.accuracy, model.loss], {model.x: train, model.y: label})
       
        print('epoch: ', e+1, " loss: ", loss, " acc: ", acc)

            
        if (e+1) % 5 ==0 or e == 0:
            for step in range( int( mnist.test.images.shape[0]  / BATCH_SIZE ) ):
                test = mnist.test.images[step * BATCH_SIZE :( step * BATCH_SIZE + BATCH_SIZE )]
                label = mnist.test.labels[step * BATCH_SIZE :( step * BATCH_SIZE + BATCH_SIZE )]

                pred = sess.run([model.prediction],{model.x: test, model.y: label})

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["result/prediction"])
    with tf.gfile.FastGFile("mnist.pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())
          
    