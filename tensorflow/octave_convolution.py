import tensorflow as tf

def octave_conv(oct_type, H = 0, L = 0, filter_num_out = 2048, kernel_size = 5, kernel = tf.keras.initializers.lecun_normal(), alpha_in = 0.5, alpha_out = 0.5):
    if oct_type == 'first':
        alpha_in = 0
        filter_num_out = filter_num_out * 2
        H_in = H[:,:,:, int(alpha_in * int(H.shape[3])):]
        
        #filter num of Low and High frequency
        L_out = int(alpha_out * filter_num_out)
        H_out = filter_num_out- int(alpha_out * filter_num_out)
        
        H2H = tf.layers.conv2d(H_in, H_out, kernel_size, 1, 'same', kernel_initializer = kernel)
        
        H2L = tf.layers.average_pooling2d(H_in, kernel_size, 2, 'same')
        H2L = tf.layers.conv2d(H2L, L_out, kernel_size, 1, 'same', kernel_initializer = kernel)
        
        H = H2H
        L = H2L
        
        return H, L


    elif oct_type == 'regular':
        
        filter_num_out = filter_num_out * 2
        L_in = L
        H_in = H
        #filter num of Low and High frequency
        L_out = int(alpha_out * filter_num_out)
        H_out = filter_num_out- int(alpha_out * filter_num_out)
        
        L2L = tf.layers.conv2d(L_in, L_out, kernel_size, 1, 'same', kernel_initializer = kernel)
        H2H = tf.layers.conv2d(H_in, H_out, kernel_size, 1, 'same', kernel_initializer = kernel)
        
        L2H = tf.layers.conv2d_transpose(L_in, H_out, kernel_size, 2, 'same', kernel_initializer = kernel)
        H2L = tf.layers.average_pooling2d(H_in, kernel_size, 2, 'same')
        H2L = tf.layers.conv2d(H2L, L_out, kernel_size, 1, 'same', kernel_initializer = kernel)
        
        H = H2H + L2H
        L = L2L + H2L
        
        return H, L
    
    else:
        alpha_out = 0
        L_in = L
        H_in = H
        #filter num of Low and High frequency
        L_out = int(alpha_out * filter_num_out)
        H_out = filter_num_out- int(alpha_out * filter_num_out)
        
        L2L = tf.layers.conv2d(L_in, L_out, kernel_size, 1, 'same', kernel_initializer = kernel)
        H2H = tf.layers.conv2d(H_in, H_out, kernel_size, 1, 'same', kernel_initializer = kernel)
        
        L2H = tf.layers.conv2d_transpose(L_in, H_out, kernel_size, 2, 'same', kernel_initializer = kernel)
        H2L = tf.layers.average_pooling2d(H_in, kernel_size, 2, 'same')
        H2L = tf.layers.conv2d(H2L, L_out, kernel_size, 1, 'same', kernel_initializer = kernel)
        
        H = H2H + L2H
        L = L2L + H2L
        
        return H


kernel = tf.keras.initializers.lecun_normal()
img = tf.zeros([10, 2000, 3000, 1024])
H, L = octave_conv('first', img, 0 , 2048, 5, kernel, 0, 0.5)
print("first "); print("H shape:", H.shape); print("L shape:", L.shape)
H, L = octave_conv('regular', H, L , 2048, 5, kernel, 0.5, 0.5)
print("regular "); print("H shape:", H.shape); print("L shape:", L.shape)
H = octave_conv('last', H, L , 2048, 5, kernel, 0.5, 0)
print("last"); print("H shape:", H.shape)

"""
first 
H shape: (10, 2000, 3000, 2048)
L shape: (10, 1000, 1500, 2048)
regular 
H shape: (10, 2000, 3000, 2048)
L shape: (10, 1000, 1500, 2048)
last
H shape: (10, 2000, 3000, 2048)
""
