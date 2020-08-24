import os
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras import Model

gpu_num = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu_num], True)

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

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer = tf.keras.optimizers.Adam(1e-3), 
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics = ['accuracy'])
                
history = model.fit(train_ds, epochs = 1000, callbacks=[tensorboard_callback])

model.save_weights('mnist_ckpt')
