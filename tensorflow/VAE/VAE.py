#read data and transfer to npy
import warnings
import os
import random
import numpy as np
import tensorflow as tf
from model import MODEL
import matplotlib.pyplot as plt
import cv2


#hyperparameter
EPOCH = 500
time_set = '0526'
BATCH_SIZE = 500
LATENT_SIZE = 4096
FILTER_NUM = 32
LR = 1e-4

os.environ['TF_CPP_MIN_LOG_LEVEL']='1' #do not show run error
os.environ['CUDA_VISIBLE_DEVICES']='0' #use which GPU device
warnings.filterwarnings('ignore')
print('This is EPOCH :', EPOCH)


#add and adjust hyperparameter
path = os.getcwd()
file = os.listdir(path + '/cartoon/')
for f in range(len(file)):
	file[f] = path + '/cartoon/'+ file[f] 
print('file : ', len(file))
test = list(file); test.sort()
print('All file name finished')


# create session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.5


save = ''
tf.reset_default_graph()
model = MODEL(LR, FILTER_NUM, BATCH_SIZE, LATENT_SIZE)
saver = tf.train.Saver(max_to_keep = 10)
with tf.Session(config=config) as sess:
    #saver.restore(sess, './AOI/model/'+'0308'+'/tf_AOI_n2n_'+'1'+'.ckpt')
    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter('TensorBoard/', sess.graph)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    #training
    print('Total steps: ' + str( int( len(file) / BATCH_SIZE ) ) )
    for e in range(EPOCH):
        sess.run(model.dataset_iter.initializer, feed_dict={model.x: file})
        
        random.shuffle(file)
        for step in range( int( len(file) / BATCH_SIZE ) ):
            _, t_loss, loss_mse, loss_pdf = sess.run([model.train_op, model.loss, model.loss_mse, model.loss_pdf])
            if step % 10  == 0:
                print('epoch: ', e+1, 'STEP: ', step, ' loss: ', t_loss, ' loss_mse: ', loss_mse, ' loss_pdf: ', loss_pdf)
                     
        try:
            os.mkdir('./model/' + time_set)
        except:
            pass
        try:
            os.mkdir('./gen_img/')
        except:
            pass
            
        if (e+1) % 10 ==0 or e == 0:
            pred = np.zeros([1, 64, 64, 3])
            sess.run(model.dataset_iter.initializer, feed_dict={model.x: test})
            for step in range( int( len(test) / BATCH_SIZE ) ):
                loss_, pred_ = sess.run([model.loss, model.output])#;print(pred_[0][27:37,27:37,0])
                print(pred_[0])
                pred_ = np.reshape(np.array(pred_), (BATCH_SIZE, 64, 64, 3))#; pred_ = pred_*127.5 + 127.5
                pred = np.concatenate([pred, pred_], axis = 0)#;print(pred[1,27:37,27:37,0])
                
            pred = pred[1:,:,:,:]#; pred = pred.astype(np.uint8)#;print(pred[0,30:34,32:34,0])
            for p in range(pred.shape[0]):
                answer = plt.imread(test[p])[88:412, 88:412, :3]
                answer= cv2.resize(answer, dsize=(64, 64))
                new = np.concatenate([answer, pred[p,:,:,:]], axis = 1)
                plt.imsave('./gen_img/' + str(p+1) + '.png', new)
                    
                
            #saver.save(sess, './model/' + time_set + '/hw3-1_' + str(e+1) + '.ckpt')
            print('Save model finish')
            
        
        with open(str(time_set)+'.txt', 'w')as f:
            f.write(save)

    #saver.save(sess, './model/' + time_set + '/hw3-1_' + str(e+1) + '.ckpt')
    print('Save model finish')
    
