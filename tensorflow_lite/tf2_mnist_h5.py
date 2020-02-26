import math

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print("x_train.shape = {}, y_train.shape = {}".format(x_train.shape, y_train.shape))
print("x_test.shape = {}, y_test.shape = {}".format(x_test.shape, y_test.shape))

inputs = Input(shape=(28, 28, 1), name='input')

x = Conv2D(24, kernel_size=(6, 6), strides=1)(inputs)
x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = Activation('relu')(x)
x = Dropout(rate=0.25)(x)

x = Conv2D(48, kernel_size=(5, 5), strides=2)(x)
x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = Activation('relu')(x)
x = Dropout(rate=0.25)(x)

x = Conv2D(64, kernel_size=(4, 4), strides=2)(x)
x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = Activation('relu')(x)
x = Dropout(rate=0.25)(x)

x = Flatten()(x)
x = Dense(200)(x)
x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = Activation('relu')(x)
x = Dropout(rate=0.25)(x)

predications = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

model = Model(inputs=inputs, outputs=predications)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

lr_decay = lambda epoch: 0.0001 + 0.02 * math.pow(1.0 / math.e, epoch / 3.0)
decay_callback = LearningRateScheduler(lr_decay, verbose=1)

history = model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, 
                    validation_data=(x_test, y_test), callbacks=[decay_callback])

model.save('mnist.h5')
converter = tf.lite.TFLiteConverter.from_keras_model_file('mnist.h5')
tflite_model = converter.convert()
open('mnist_tf2_saveh5.tflite', 'wb').write(tflite_model)
