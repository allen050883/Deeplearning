import os
import pathlib
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

mnist_0 = [os.path.join(os.getcwd(), 'mnist', '0', i) for i in os.listdir("./mnist/0/")]
mnist_1 = [os.path.join(os.getcwd(), 'mnist', '1', i) for i in os.listdir("./mnist/1/")]
mnist_2 = [os.path.join(os.getcwd(), 'mnist', '2', i) for i in os.listdir("./mnist/2/")]
mnist_3 = [os.path.join(os.getcwd(), 'mnist', '3', i) for i in os.listdir("./mnist/3/")]
mnist_4 = [os.path.join(os.getcwd(), 'mnist', '4', i) for i in os.listdir("./mnist/4/")]
mnist_5 = [os.path.join(os.getcwd(), 'mnist', '5', i) for i in os.listdir("./mnist/5/")]
mnist_6 = [os.path.join(os.getcwd(), 'mnist', '6', i) for i in os.listdir("./mnist/6/")]
mnist_7 = [os.path.join(os.getcwd(), 'mnist', '7', i) for i in os.listdir("./mnist/7/")]
mnist_8 = [os.path.join(os.getcwd(), 'mnist', '8', i) for i in os.listdir("./mnist/8/")]
mnist_9 = [os.path.join(os.getcwd(), 'mnist', '9', i) for i in os.listdir("./mnist/9/")]

train = mnist_0 + mnist_1 + mnist_2 + mnist_3 + mnist_4 + mnist_5 + mnist_6 + mnist_7 + mnist_8 + mnist_9
label = list(np.concatenate([0*np.ones(len(mnist_0)), 1*np.ones(len(mnist_1)), 2*np.ones(len(mnist_2)), 3*np.ones(len(mnist_3)), 4*np.ones(len(mnist_4)), 5*np.ones(len(mnist_5)), 6*np.ones(len(mnist_6)), 7*np.ones(len(mnist_7)), 8*np.ones(len(mnist_8)), 9*np.ones(len(mnist_9))]))

def img_preprocess(x):
    return tf.cast( tf.reshape( tf.io.decode_jpeg( tf.io.read_file(x)), (28, 28, 1)), tf.float32)
def normalize(x):
    return (x - 127.5)/127.5

train_ds = tf.data.Dataset.from_tensor_slices((train, label))
train_ds = train_ds.map(lambda x, y: (normalize(img_preprocess(x)), y))
train_ds = train_ds.batch(8192, drop_remainder = True)
train_ds = train_ds.prefetch( buffer_size = 8192 )  # prefech

#test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, name = 'prob')

    def call(self, x):
        return self.CNN_layer(x)
      
    def CNN_layer(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
        
    

# Create an instance of the model
model = MyModel()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(optimizer = tf.keras.optimizers.RMSprop(), 
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics = ['sparse_categorical_accuracy'])
                
history = model.fit(train_ds, epochs=3, callbacks = [callback])


#save pb
tf.saved_model.save(model, "./mnist_tf2/")
converter = tf.lite.TFLiteConverter.from_saved_model("./mnist_tf2/")
tflite_model = converter.convert()
open('mnist_tf2_savepb.tflite', 'wb').write(tflite_model)
