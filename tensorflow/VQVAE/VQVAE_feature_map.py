import os
import cv2
import pickle
import random
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from VQVAE_model_feature_map import MODEL
from tensorflow.examples.tutorials.mnist import input_data

#hyperparameter
EPOCH = 50000
time_set = '0813'
BATCH_SIZE = 20
FILTER_NUM = 32
BETA = 0.25
LR = 1e-6
K = 128


os.environ['TF_CPP_MIN_LOG_LEVEL']='1' #do not show run error
os.environ['CUDA_VISIBLE_DEVICES']='0' #use which GPU device
warnings.filterwarnings('ignore')
print('This is EPOCH :', EPOCH)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train = mnist.train.images; train = np.reshape(train, (train.shape[0], 28, 28, 1));train = train[:1000,:,:,:]
train_label = mnist.train.labels
test = mnist.test.images; test = np.reshape(test, (test.shape[0], 28, 28, 1));test = test[:20,:,:,:]
test_label = mnist.test.labels

# create session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.5


save = ''
tf.reset_default_graph()
model = MODEL(LR, FILTER_NUM, BETA, K, True, BATCH_SIZE)
#VQVAE(1e-4, 32 ,0.1, 20, True, 500, 10)
saver = tf.train.Saver(max_to_keep = 10)
global_step = 0
with tf.Session(config=config) as sess:
    #saver.restore(sess, './AOI/model/'+'0308'+'/tf_AOI_n2n_'+'1'+'.ckpt')
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('TensorBoard/', sess.graph)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    #training
    print('Total steps: ' + str( int( len(train) / BATCH_SIZE ) ) )
    for e in range(EPOCH):
        sess.run(model.dataset_iter.initializer, feed_dict={model.x: train})
        #rand_int = random.randint(0, 2019)
        #random.seed(rand_int); random.shuffle(train)
        #random.seed(rand_int); random.shuffle(test)
        for step in range( int( train.shape[0] / BATCH_SIZE ) ):
            _, loss, pred, recon, vq, commit = sess.run([model.train_op, model.loss, model.output, model.recon, model.vq, model.commit])
            global_step += 1
            
            if step % 10  == 0:
                print('epoch: ', e+1, 'STEP: ', step, ' loss: ', loss, 'recon: ', recon, 'vq: ', vq, 'commit: ', commit)
                     
        try:
            os.mkdir('./model/' + time_set)
        except:
            pass
        try:
            os.mkdir('./gen_img/')
        except:
            pass
            
        if (e+1) % 5 ==0 or e == 0:
            #pred = np.zeros([1, 28, 28, 1])
            sess.run(model.dataset_iter.initializer, feed_dict={model.x: test})
            loss_, pred_ = sess.run([model.loss, model.output])#;print(pred_[0][27:37,27:37,0])
            pred_ = np.reshape(np.array(pred_), (-1, 28, 28, 1))#; pred_ = pred_*127.5 + 127.5
            #pred = np.concatenate([pred, pred_], axis = 0)#;print(pred[1,27:37,27:37,0])
                
            #pred = pred[1:,:,:,:]#; pred = pred.astype(np.uint8)#;print(pred[0,30:34,32:34,0])
            for p in range(pred_.shape[0]):
                if p < 10:
                    answer = test[p,:,:,:]
                    answer = answer * 255.
                    pred_img = pred_[p,:,:,:] * 255.
                    new = np.concatenate([answer, pred_img], axis = -2)
                    cv2.imwrite('./gen_img/' + str(p+1) + '.jpg', new)
                    
                
            #saver.save(sess, './model/' + time_set + '/hw3-1_' + str(e+1) + '.ckpt')
            print('Save model finish')
            
        
        with open(str(time_set)+'.txt', 'w')as f:
            f.write(save)

    #saver.save(sess, './model/' + time_set + '/hw3-1_' + str(e+1) + '.ckpt')
    print('Save model finish')