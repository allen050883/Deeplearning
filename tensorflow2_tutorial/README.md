## tf2 and tf.keras  
#### 1. Sessions and eager execution  
The model do not have to build the computation graph, it is easy to debug.  
#### 2. Automatic differentiation  
GradientTape and mode.fit to help automatic differentiation.  
#### 3. Model and layer subclassing  
Sequential  
Function  
Subclassing  
Both the sequential and functional paradigms have been inside Keras for quite a while, but the subclassing feature is still unknown to many deep learning practitioners.  
```
# Subclassing
class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=6, 
                           kernel_size=(3, 3), activation='relu', 
                           input_shape=(32,32,1))
        self.average_pool = tf.keras.layers.AveragePooling2D()
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=16, 
                           kernel_size=(3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(84, activation='relu')
        self.out = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, input):
        x = self.conv2d_1(input)
        x = self.average_pool(x)
        x = self.conv2d_2(x)
        x = self.average_pool(x)
        x = self.flatten(x)
        x = self.fc_2(self.fc_1(x))
        return self.out(x)
    
lenet = LeNet()
```
#### 4. Better multi-GPU/distributed training support  

image source: https://jhui.github.io/2017/03/07/TensorFlow-GPU/  

## Tensorboard in tf2  
  
```python
import datetime
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])
```
  
For command line,  
```bash
tensorboard --logdir logs/fit --host IP --port 6006
```
