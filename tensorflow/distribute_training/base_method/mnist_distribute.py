from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder(tf.float32,[None,784])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder("float",[None,10])

cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


BATCH=100
with tf.Session("grpc://10.0.0.1:10000") as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(1000):
        print("EPOCH: ", epoch)
        for i in range(int(50000/BATCH)):
            batch_xs,batch_ys=mnist.train.next_batch(BATCH)
            sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
