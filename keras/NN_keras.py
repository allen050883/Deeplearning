import numpy as np

#Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((-1, 28*28))/256
X_test = X_test.reshape((-1, 28*28))/256

y_train_1hot = np.zeros((y_train.shape[0], 10))
for i in range(y_train_1hot.shape[0]):
    y_train_1hot[i, y_train[i]] = 1

y_test_1hot = np.zeros((y_test.shape[0], 10))
for i in range(y_test_1hot.shape[0]):
    y_test_1hot[i, y_test[i]] = 1



from keras.models import Sequential
model = Sequential()

from keras.layers import Dense, Activation
model.add(Dense(output_dim=256, input_dim=784))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train_1hot, nb_epoch=5, batch_size=32)
loss_and_metrics = model.evaluate(X_test, y_test_1hot, batch_size=32)

print(loss_and_metrics)
