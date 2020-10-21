## tf2 and tf.keras  
#### 1. Sessions and eager execution  
The model do not have to build the computation graph, it is easy to debug.  
#### 2. Automatic differentiation  

Model and layer subclassing
Better multi-GPU/distributed training support

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
