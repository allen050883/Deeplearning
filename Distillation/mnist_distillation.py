import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D,GlobalAveragePooling2D,Dense,Softmax,Flatten,MaxPooling2D,Dropout,Activation, Lambda, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import kullback_leibler_divergence as KLD_Loss, categorical_crossentropy as logloss
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import categorical_accuracy
import seaborn as sns

from solve_cudnn_error import *
solve_cudnn_error()

NUM_CLASSES = 10
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]     
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)


y_train_not_onehot = y_train
y_test_not_onehot  = y_test
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
 
# Normalize the dataset
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

Teacher = Sequential() 
Teacher.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32,32,3)))
Teacher.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
Teacher.add(MaxPooling2D(pool_size=2))
Teacher.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
Teacher.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
Teacher.add(MaxPooling2D(pool_size=2))
Teacher.add(Dropout(0.5))
Teacher.add(Flatten())
Teacher.add(Dense(512, activation='relu'))
Teacher.add(Dropout(0.5))
Teacher.add(Dense(10))
Teacher.add(Activation('softmax'))

Teacher.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
 
Teacher.summary()


myCP = ModelCheckpoint(save_best_only=True,filepath='teacher.h5',monitor = 'accuracy')
Teacher.fit(x_train,
         y_train,
         batch_size=1024,
         epochs=100,
         validation_split = 0.2,
         callbacks=[myCP])

Teacher = load_model('teacher.h5')
#用test dataset評估結果
Teacher.evaluate(x_test,y_test)

Teacher_logits = Model(Teacher.input,Teacher.layers[-2].output)

logits_plot = []              
#取training dataset第七張圖片來作處理
item_idx = 7
item_image = x_train[item_idx]
plt.imshow(item_image)
plt.savefig('horse.jpg')
plt.close()


Temperatures = [1,5,10,20,35,50]
for Temperature in Temperatures:
    # 將teacher model的輸出x除上Temperature
    T_layer = Lambda(lambda x:x/Temperature)(Teacher_logits.output)
    
    #手動建一softmax layer，使用上述的temperature
    Softmax_layer = Softmax()(T_layer)
    #用上面的方式，把Softmax_layer放入模型中
    Teacher_logits_soften = Model(Teacher.input,Softmax_layer)
                                
    # Append for plotting
    logits_plot.append(Teacher_logits_soften.predict(np.array([item_image])))
    plt.figure(figsize=(14, 6))

 
for i in range(len(Temperatures)):
    sns.lineplot(class_names,logits_plot[i][0],legend="full")
    plt.title('This is a '+ class_names[y_train_not_onehot[item_idx][0]])
    plt.legend(Temperatures,title="Temperatures")
plt.savefig('temp.jpg')
plt.close()



############################Student Net#######################################
Student = Sequential() #a Must define the input shape in the first layer of the neural network
Student.add(Flatten(input_shape=(32,32,3)))
Student.add(Dense(64, activation='relu'))
Student.add(Dense(10))
Student.summary()

student_logits = Student.layers[-1].output
probs = Activation("softmax")(student_logits)
logits_T = Lambda(lambda x:x/Temperature)(student_logits)
probs_T = Activation("softmax")(logits_T)
CombinedLayers = concatenate([probs,probs_T])
 
StudentModel = Model(Student.input,CombinedLayers)
StudentModel.summary()


def KD_loss(y_true, y_pred, lambd=0.5, T=10.0):
    y_true_KD = y_true
    y_pred,y_pred_KD = y_pred[:,:NUM_CLASSES],y_pred[:,NUM_CLASSES:]
    # Classic cross-entropy (without temperature)
    CE_loss = tf.keras.losses.CategoricalCrossentropy()
    # KL-Divergence loss for softened output (with temperature)
    KLD = tf.keras.losses.KLDivergence()
    KL_loss = T**2*KLD(y_true_KD, y_pred_KD)
    return lambd*CE_loss(y_true,y_pred) + (1-lambd)*KL_loss
    
def accuracy(y_true,y_pred):
    return categorical_accuracy(y_true,y_pred)


StudentModel.compile(loss=KD_loss, optimizer='adam', metrics=['accuracy'])

myCP = ModelCheckpoint(save_best_only=True, filepath='student_10.h5',monitor = 'accuracy')
StudentModel.fit(x_train, y_train, epochs=50, validation_split=0.2, batch_size=1024, callbacks=[myCP])

StudentModel.load_weights('student_10.h5')
StudentModel.evaluate(x_test, y_test)