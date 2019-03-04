import tensorflow as tf 

def Attention(Q, K, name='att'):
    
    Qf, Kf = tf.layers.flatten(Q), tf.layers.flatten(K)
    attention_map = tf.reshape(Kf,[tf.shape(Kf)[0], tf.shape(Kf)[1], 1]) * tf.reshape(Qf,[tf.shape(Qf)[0], 1, tf.shape(Qf)[1]]) 
    attention_map = tf.nn.softmax(attention_map, axis=1) # consider only the keys for attention
    attention = tf.reshape(Kf,[tf.shape(Kf)[0], tf.shape(Kf)[1], 1]) * attention_map 
    attention = tf.reduce_sum(attention, axis=1)
    gamma = tf.get_variable(name+"att_gamma", [1], initializer=tf.constant_initializer(0.0)) # set the gamma as learnable variable

    return tf.reshape(attention, tf.shape(Q)) * tf.nn.sigmoid(gamma) + Q * tf.nn.sigmoid(1 - gamma) # V

# use difference source and target 
S = tf.zeros([10, 28, 28, 3])
T = tf.zeros([10, 14, 14, 1024])
A = Attention(T, S, 'att1')
print(A) # output should have the same shape of T (10, 14, 14, 1024)

# puting self as two inputs will be self-attention
A = Attention(T, T, 'att2')
print(A)
