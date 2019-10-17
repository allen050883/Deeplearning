import os
import tensorflow as tf
import math
import numpy as np
from functools import partial
import VQVAE_ema_module

class CoordConv2D:
    def __init__(self, with_r = False):
        self.with_r = with_r
    def __call__(self,input):
        self.x_dim = input.shape.as_list()[2]
        self.y_dim = input.shape.as_list()[1]
        batch_size_tensor = tf.shape(input)[0]
        xy_vector = tf.ones([self.y_dim,1])
        yx_vector = tf.ones([1,self.x_dim])
        x_range = tf.reshape(tf.range(1,self.x_dim+1,1,dtype=tf.float32),[1,self.x_dim])
        y_range = tf.reshape(tf.range(1,self.y_dim+1,1,dtype=tf.float32),[self.y_dim,1])
        x_normal_range = tf.multiply(x_range,1/self.x_dim)
        y_normal_range = tf.multiply(y_range,1/self.y_dim)
        x_mat = tf.matmul(xy_vector,x_normal_range)
        y_mat = tf.matmul(y_normal_range,yx_vector)

        x_mat = tf.reshape(x_mat,[1,self.y_dim,self.x_dim,1])
        y_mat = tf.reshape(y_mat,[1,self.y_dim,self.x_dim,1])
        x_mats = tf.tile(x_mat,[batch_size_tensor,1,1,1])
        y_mats = tf.tile(y_mat,[batch_size_tensor,1,1,1])


        
        if self.with_r == True:
            # # orgin
            # r = ((x_mats-0.5)**2 + (y_mats-0.5)**2)
            # r = tf.sqrt(r)

            # I test 
            r = (tf.sqrt((x_mats-0.5)**2) + tf.sqrt((y_mats-0.5)**2))

            input = tf.concat([input,x_mats,y_mats,r],axis=-1)
            return input
        else:
            input = tf.concat([input,x_mats,y_mats],axis=-1)
            return input


class MODEL:
    defalt_data_img_holder = tf.placeholder(tf.float32, [None, 72 * 2, 128 * 2, 3], "default_data_img_holder")

    def __init__(self, LR, filter_num, batch_size, latent_size, latent_base, attention_head_num):
        self.LR = LR
        self.filter_num = filter_num
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.latent_base = latent_base
        self.kernel = tf.keras.initializers.glorot_normal()
        self.attention_head_num = attention_head_num

        global_step = tf.Variable(0, trainable=False)

        # learning_rate = tf.train.exponential_decay(self.LR, global_step, 10000, 0.96, staircase=True)
        learning_rate = self.LR

        # place holders
        # self.ori_x = tf.placeholder(tf.string, [None])
        self.ori_x = tf.placeholder(tf.float32, [None, 28,28],name="ori_x_holder")

        self.ori_y = tf.placeholder(tf.string, [None])
        self.keep_training = tf.placeholder_with_default(True, shape=())
        # data_img_holder = tf.placeholder(tf.float32, [None, 28, 28, 1], "default_data_img_holder")
        # self.data_img_holder = tf.multiply(data_img_holder,1/255,"input_regularize")
        self.data_img_holder = tf.placeholder(tf.float32, [None, 28, 28, 1], "default_data_img_holder")
        self.reg_data_img_holder = self.data_img_holder/255

        dataset = tf.data.Dataset.from_tensor_slices(self.ori_x)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10))
        dataset = dataset.prefetch(10)
        self.dataset_iter = dataset.make_initializable_iterator()
        self.dataset_prefetch = tf.reshape(self.dataset_iter.get_next(),[-1, 28, 28, 1])

        self.gray_data_img = self.reg_data_img_holder

        self.main(self.gray_data_img, training_status=self.keep_training)

        self.loss = self.loss_function()

        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=.9, centered=True)

        self.train_op = optimizer.minimize(self.loss)

        # built reconstruct_img
        

    def short_cut_layer(self, enc_layer, dec_layer):
        short_cut1_H_x = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="SAME",
                                                kernel_initializer=tf.keras.initializers.glorot_normal())(enc_layer)

        short_cut1_H_x = tf.keras.layers.Conv2D(dec_layer.shape.as_list()[-1], kernel_size=3, strides=1, padding="SAME",
                                                kernel_initializer=tf.keras.initializers.glorot_normal())(
            short_cut1_H_x)

        expand_dec_layer = tf.concat([short_cut1_H_x, dec_layer], axis=3)

        return expand_dec_layer

    def main(self, img, training_status):
        def lrelu(x, alpha=0.3):
            return tf.maximum(x, tf.multiply(x, alpha))
    

        with tf.variable_scope("vea_autoencoder", reuse=tf.AUTO_REUSE):
            activation = lrelu
            with tf.variable_scope("encoder", reuse=None):
                X = tf.reshape(img, shape=[-1, 28, 28, 1])
                x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.tanh)
                x = tf.nn.dropout(x, 0.8)
                x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.tanh)
                x = tf.nn.dropout(x, 0.8)
                l3_output = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.tanh)

                print("l3_output:", l3_output)

            img_shape = l3_output.shape


            with tf.variable_scope("top_VQVAE"):
                top_VQVAE_instance = VQVAE_ema_module.VQVAE(l3_output, self.latent_size, 0.25, "top_VQVAE")
                top_VQ_out_dict = top_VQVAE_instance.VQVAE_layer_out()

            top_VQ_out = top_VQ_out_dict['quantized_embd_out']
            self.top_VQ_loss = top_VQ_out_dict["VQ_loss"]
            self.top_VQ_encodings = top_VQ_out_dict["encodings"]
            self.top_VQ_assign_moving_avg_op = top_VQ_out_dict['assign_moving_avg_op']
            self.top_VQ_temp_decay_op = top_VQ_out_dict["temp_decay_op"]
            self.top_k_idx = top_VQ_out_dict["top_k_idx"]
            # self.verf_dist_same = top_VQVAE_instance.verf_dist_same

            # print("top_VQ_out:", top_VQ_out)

            # unflatten_ouput = VQ_out
            #
            # print("unflatten_ouput:", unflatten_ouput)

            channel_reconstruct = tf.keras.layers.Dense(img_shape[-1],
                                                        kernel_initializer=tf.keras.initializers.glorot_normal())(
                top_VQ_out)
            print("channel_reconstruct:", channel_reconstruct)

            x = tf.layers.conv2d_transpose(channel_reconstruct, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.tanh)
            x = tf.nn.dropout(x, 0.8)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.tanh)
            x = tf.nn.dropout(x, 0.8)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.tanh)

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=28*28, activation=tf.nn.sigmoid)
            recon_out = tf.reshape(x, shape=[-1, 28, 28,1])

            # reconstruct layer

            # recon_out = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="SAME", activation="sigmoid",
            #                                    kernel_initializer=self.kernel)(l5_output)
            # recon_out = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding="SAME",
            #                                    kernel_initializer=self.kernel, activation="sigmoid")(recon_out)
            self.reg_recon_out = recon_out
            self.recon_output = tf.cast(tf.round(recon_out * 255), tf.float32)

        # print("seg_map:", seg_map.shape)
        # print(("recon_out:", recon_out))

        

    def loss_function(self):
        # reconstruct loss
        # data_img = (self.data_img + 1) * .5  # from -1~1 to 0~1

        # print("data_img:",data_img)
        print("self.recon_output:",self.recon_output)

        self.reconstruct_loss = tf.reduce_mean(tf.squared_difference(self.reg_data_img_holder, self.reg_recon_out), axis=[1, 2, 3])

        self.logpx_z = self.reconstruct_loss

        return tf.reduce_mean(self.logpx_z)  + self.top_VQ_loss

    def oct_conv_first_layer(self, x, channel_num, alpha, kernel_size=3, activation=tf.nn.tanh):
        H_channel_num = int(channel_num * alpha // 1)  # by alpha, I split channel to high freq and low freq chuncks
        L_channel_num = channel_num - H_channel_num

        H_x = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.glorot_normal(),
                                     activation=activation)(x)

        # since low freq catch the spatial stucture rather than catching detail, we use pooling on Low freq parts
        L_pooling = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')(x)
        L_x = tf.keras.layers.Conv2D(L_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.glorot_normal(),
                                     activation=activation)(L_pooling)

        return H_x, L_x

    def oct_conv_block(self, H_x, L_x, channel_num, alpha, kernel_size=3, activation=tf.nn.tanh):
        H_channel_num = int(channel_num * alpha // 1)  # by alpha, I split channel to high freq and low freq chuncks
        L_channel_num = channel_num - H_channel_num

        H2H = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.glorot_normal())(H_x)

        # # dilation add-on
        # H2dilation = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
        #                                     dilation_rate=2,
        #                                     kernel_initializer=tf.keras.initializers.glorot_normal())(H_x)
        # H2H = tf.concat([H2H, H2dilation], axis=3)
        # H2H = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
        #                              kernel_initializer=tf.keras.initializers.glorot_normal())(H2H)

        H2L = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')(H_x)
        H2L = tf.keras.layers.Conv2D(L_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.glorot_normal())(H2L)

        L2L = tf.keras.layers.Conv2D(L_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.glorot_normal())(L_x)

        # upsampling to H freq parts size
        L2H_raw = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(L_x)
        L2H = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.glorot_normal())(L2H_raw)

        # # dilation add-on
        # L2dilation = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
        #                                     dilation_rate=2,
        #                                     kernel_initializer=tf.keras.initializers.glorot_normal())(L2H_raw)
        # L2H = tf.concat([L2H, L2dilation], axis=3)
        # L2H = tf.keras.layers.Conv2D(H_channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
        #                              kernel_initializer=tf.keras.initializers.glorot_normal())(L2H)

        return activation((H2H + L2H) / 2), activation((L2L + H2L) / 2)

    def oct_conv_final_layer(self, H_x, L_x, channel_num, kernel_size=3, activation=tf.nn.tanh):
        L2H = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(L_x)
        L2H = tf.keras.layers.Conv2D(channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.glorot_normal())(L2H)
        H2H = tf.keras.layers.Conv2D(channel_num, kernel_size=kernel_size, strides=1, padding="SAME",
                                     kernel_initializer=tf.keras.initializers.glorot_normal())(H_x)

        return activation((H2H + L2H) / 2)
