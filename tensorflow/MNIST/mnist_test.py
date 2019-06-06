import tensorflow as tf
import  numpy as np
from skimage import io
import os

#def test_data_file():
#    """just load other data"""
#    X_train = np.zeros([1, 28*28]); Y_train = np.zeros([1])
#    if os.path.exists("./test_data/"):
#        file_all = os.listdir("./test_data/")
#        for f_a in file_all:
#            file = os.listdir("./test_data/"+f_a+"/")
#            print(f_a+" has "+str(len(file))+" files")
#            for f in file:
#                test = io.imread("./test_data/"+f_a+"/"+f)
#                test = test.astype(np.float32)
#                if len(test.shape) > 2:
#                    test = test[:,:,0]
#                test = np.reshape(test, (1, 28*28))
#                X_train = np.concatenate([X_train, test], axis = 0) 
#                Y_train = np.concatenate([Y_train, np.array([int(f_a)])])
#        print("X_train and Y_train concatenate finish!")
#    else:
#        print("There is no test_data file.")
#    return X_train[1:,:], Y_train[1:]



with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open("mnist.pb", "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name = "")
        

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        input_x = sess.graph.get_tensor_by_name("model_input/input:0")
        softmax = sess.graph.get_tensor_by_name("result/softmax:0")
        #label = sess.graph.get_tensor_by_name("import/input/label")
        pred = sess.graph.get_tensor_by_name("result/prediction:0")
        #acc = sess.graph.get_tensor_by_name("import/result/accuracy")
        
       
        img = io.imread("44.png"); img = img.astype(np.float32)
        if len(img.shape) > 2:
            img = img[:,:,0]
        img = np.reshape(img, (1, img.shape[0]*img.shape[1]))
        pred_num, soft = sess.run([pred, softmax], feed_dict={input_x: img})
        print(pred_num)
        print(soft)
        
#%%
"""
with tf.gfile.GFile("mnist.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# import graph_def
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name = "")

# print operations
for op in graph.get_operations():
    print(op.name)
"""        
