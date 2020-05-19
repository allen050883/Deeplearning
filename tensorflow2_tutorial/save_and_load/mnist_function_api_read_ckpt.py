import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

#structure
input = Input(shape=(28, 28, 1), name = 'input_data')
x = Conv2D(32, 3, activation = tf.nn.relu, name = 'conv1')(input)
x = Conv2D(64, 3, activation = tf.nn.relu, name = 'conv2')(x)
x = Flatten()(x)
x = Dense(128, activation = tf.nn.relu, name = 'dense1')(x)
output = Dense(10, name = 'output', activation = tf.nn.sigmoid)(x)

model = Model(inputs = input, outputs = [output], name = 'just_model')  
model.load_weights('mnist_ckpt')

prediction = model.predict(test_ds)
number = np.argmax(prediction, axis = -1)
print(number)
