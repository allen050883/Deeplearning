import tensorflow as tf 

def Attention(Q, K, name='att'):
    
    Qf, Kf = tf.layers.flatten(Q), tf.layers.flatten(K)
    attention_map = tf.reshape(Kf,[tf.shape(Kf)[0], tf.shape(Kf)[1], 1]) * tf.reshape(Qf,[tf.shape(Qf)[0], 1, tf.shape(Qf)[1]]) 
    attention_map = tf.nn.softmax(attention_map, axis=1) # consider only the keys for attention
    attention = tf.reshape(Kf,[tf.shape(Kf)[0], tf.shape(Kf)[1], 1]) * attention_map 
    attention = tf.reduce_sum(attention, axis=1)
    gamma = tf.get_variable(name+"att_gamma", [1], initializer=tf.constant_initializer(0.0)) # set the gamma as learnable variable

    return tf.reshape(attention, tf.shape(Q)) * tf.nn.sigmoid(gamma) + Q * tf.nn.sigmoid(1 - gamma) # V

def Spatial_attention(Q, K, compression_channel_no = 16, name = 'satt'):

    # the query and key can be firstly transform to the same channel (This can also compress the information).
    Qm = tf.layers.conv2d(Q, compression_channel_no, [1,1], [1,1], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.keras.initializers.orthogonal(), name=name+'att_compressionQ')
    Km = tf.layers.conv2d(K, compression_channel_no, [1,1], [1,1], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.keras.initializers.orthogonal(), name=name+'att_compressionK')
    V  = tf.layers.conv2d(K, Q.shape[3].value, [1,1], [1,1], padding="SAME", activation=tf.nn.elu, kernel_initializer=tf.keras.initializers.orthogonal(), name=name+'att_V')
    
    Qf = tf.reshape(Qm, [-1, Qm.shape[1].value * Qm.shape[2].value, compression_channel_no])
    Kf = tf.reshape(Km, [-1, Km.shape[1].value * Km.shape[2].value, compression_channel_no])
    Vf = tf.reshape(V, [-1,  V.shape[1].value  * V.shape[2].value, Q.shape[3].value])
    
    attention_map = tf.matmul(Qf, Kf, transpose_b=True)  # [bs, N, N]
    attention_map = tf.nn.softmax(attention_map, axis=1) # consider only the keys for attention
    attention = tf.matmul(attention_map, Vf)
    
    gamma = tf.get_variable(name+"satt_gamma", [1], initializer=tf.constant_initializer(0.0)) # set the gamma as learnable variable
    return tf.reshape(attention, tf.shape(Q)) * tf.nn.sigmoid(gamma) + Q * tf.nn.sigmoid(1 - gamma) 


def Channel_attention(Q, name = 'satt'):
    
    Qf = tf.reshape(Q, [-1, Q.shape[3].value, Q.shape[1].value * Q.shape[2].value])
    Kf = tf.reshape(Q, [-1, Q.shape[3].value, Q.shape[1].value * Q.shape[2].value])
    Vf = tf.reshape(Q, [-1, Q.shape[3].value,Q.shape[1].value * Q.shape[2].value])
    
    attention_map = tf.matmul(Qf, Kf, transpose_b=True)  # [bs, N, N]
    attention_map = tf.nn.softmax(attention_map, axis=1) # consider only the keys for attention
    attention = tf.matmul(attention_map, Vf)
    
    gamma = tf.get_variable(name+"satt_gamma", [1], initializer=tf.constant_initializer(0.0)) # set the gamma as learnable variable
    return tf.reshape(attention, tf.shape(Q)) * tf.nn.sigmoid(gamma) + Q * tf.nn.sigmoid(1 - gamma) 


# use difference source and target 
S = tf.zeros([10, 28, 28, 3])
T = tf.zeros([10, 14, 14, 1024])
A = Attention(T, S, 'att1')
print('Attention1', A) # output should have the same shape of T (10, 14, 14, 1024)

# puting self as two inputs will be self-attention
A = Attention(T, T, 'att2')
print('Attention2', A)

#spatial attention
S = tf.placeholder(tf.float32, [None,28,28,3])
T = tf.placeholder(tf.float32, [None,10,10,1024])
A = Spatial_attention(T,S) #(None, 10, 10, 1024)
print('Spatial_attention', A)

A = Channel_attention(S, 'att3'); print('Channel_attention1', A)
A = Channel_attention(T, 'att4'); print('Channel_attention2', A)
