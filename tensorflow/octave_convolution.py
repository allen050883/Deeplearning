import tensorflow as tf

def octave_conv(T, filter_num_out, kernel_size, alpha_in, alpha_out):
    L_in = T[:,:,:,: int(alpha_in * int(T.shape[3]))]
    L_in = tf.image.resize_images(L_in, (int(int(T.shape[1])/2), int(int(T.shape[2])/2)), method = 0)
    
    H_in = T[:,:,:, int(alpha_in * int(T.shape[3])):]
    
    #filter num of Low and High frequency
    L_out = int(alpha_out * filter_num_out)
    H_out = filter_num_out- int(alpha_out * filter_num_out)
    
    L2L = tf.layers.conv2d(L_in, L_out, kernel_size, 1, 'same')
    H2H = tf.layers.conv2d(H_in, H_out, kernel_size, 1, 'same')
    
    L2H = tf.layers.conv2d_transpose(L_in, H_out, kernel_size, 2, 'same')
    H2L = tf.layers.average_pooling2d(H_in, kernel_size, 2, 'same')
    H2L = tf.layers.conv2d(H2L, L_out, kernel_size, 1, 'same')
    
    H = H2H + L2H
    L = L2L + H2L
    
    return H, L

T = tf.zeros([10, 14, 14, 1024])
H, L = octave_conv(T, 2048, 3, 0.5, 0.5)
print(H.shape)
print(L.shape)
