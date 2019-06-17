import tensorflow as tf

def octave_conv(oct_type, H, L, filter_num_out, kernel_size = 3, kernel = tf.keras.initializers.lecun_normal(), activation = tf.nn.selu, alpha = 0.5):
    if oct_type == 'first':
        #filter num of Low and High frequency
        L_out = int(alpha * filter_num_out // 1)
        H_out = filter_num_out- int(alpha * filter_num_out)
        H2H = tf.keras.layers.Conv2D(H_out, kernel_size, 1, 'same', kernel_initializer = kernel, activation = activation)(H)
        H2L = tf.keras.layers.AveragePooling2D(2, 2, 'same')(H)
        H2L = tf.keras.layers.Conv2D(L_out, kernel_size, 1, 'same', kernel_initializer = kernel, activation = activation)(H2L)
        return H2H, H2L

    elif oct_type == 'regular':
        #filter num of Low and High frequency
        L_out = int(alpha * filter_num_out // 1)
        H_out = filter_num_out- int(alpha * filter_num_out)
        H2H = tf.keras.layers.Conv2D(H_out, kernel_size, 1, 'same', kernel_initializer = kernel, activation=None)(H)
        H2L = tf.keras.layers.AveragePooling2D(2, 2, 'same')(H)
        H2L = tf.keras.layers.Conv2D(L_out, kernel_size, 1, 'same', kernel_initializer = kernel, activation=None)(H2L)
        L2L = tf.keras.layers.Conv2D(L_out, kernel_size, 1, 'same', kernel_initializer = kernel, activation=None)(L)
        L2H = tf.keras.layers.Conv2D(H_out, kernel_size, 1, 'same', kernel_initializer = kernel, activation=None)(L)
        L2H = tf.keras.layers.UpSampling2D(interpolation='bilinear')(L2H)
        return activation((H2H + L2H)/2), activation((L2L + H2L)/2)
    
    else:
        #filter num of Low and High frequency
        H2H = tf.keras.layers.Conv2D(filter_num_out, kernel_size, 1, 'same', kernel_initializer = kernel, activation=None)(H)
        L2H = tf.keras.layers.Conv2D(filter_num_out, kernel_size, 1, 'same', kernel_initializer = kernel, activation=None)(L)
        L2H = tf.keras.layers.UpSampling2D(interpolation='bilinear')(L2H)
        return activation((H2H + L2H)/2)


kernel = tf.keras.initializers.lecun_normal()
img = tf.zeros([10, 32, 32, 3])
H, L = octave_conv('first', img, 0 , 64, kernel_size = 3, alpha = 0.8)
print("first "); print("H shape:", H.shape); print("L shape:", L.shape)
H, L = octave_conv('regular', H, L , 128, kernel_size = 3, alpha = 0.8)
print("regular "); print("H shape:", H.shape); print("L shape:", L.shape)
H = octave_conv('last', H, L , 256, kernel_size = 3, alpha = 0.8)
print("last"); print("H shape:", H.shape)
