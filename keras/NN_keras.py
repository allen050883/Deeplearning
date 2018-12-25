#numpy是python支援的數學函數庫
import numpy as np

#Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
from keras.datasets import mnist
#將input跟output的train跟test分別丟進X跟Y裏面
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#灰階：還是黑白的，但由最黑到最白之間可有256種明亮度
X_train = X_train.reshape((-1, 28*28))/256
X_test = X_test.reshape((-1, 28*28))/256
#np.zero代表設定全部都是0
y_train_1hot = np.zeros((y_train.shape[0], 10))
#print(y_train[1:10])     
#為output的解答，分別是0~9，將他拆成下面這樣
#ex. 0=1 0 0 0 0 0 0 0 0 0
#ex. 1=0 1 0 0 0 0 0 0 0 0
#ex. 2=0 0 1 0 0 0 0 0 0 0
for i in range(y_train_1hot.shape[0]):
    y_train_1hot[i, y_train[i]] = 1

y_test_1hot = np.zeros((y_test.shape[0], 10))
for i in range(y_test_1hot.shape[0]):
    y_test_1hot[i, y_test[i]] = 1



from keras.models import Sequential
model = Sequential()

from keras.layers import Dense, Activation
#output_dim為hidden layer    input_dim為input layer
model.add(Dense(output_dim=256, input_dim=784))
model.add(Activation("sigmoid"))
#output_dim  是output_layer  原則上是要從上層的256接下來(Tensorflow有)
model.add(Dense(output_dim=10))
#output十個數字
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train_1hot, nb_epoch=5, batch_size=32)
#nb_epoch=5做五次，batch_size分群去做
loss_and_metrics = model.evaluate(X_test, y_test_1hot, batch_size=32)

print(loss_and_metrics)
