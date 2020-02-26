# Tensorflow Lite  
### Tf1 (tf 1.14)  
Solution: "pb file" transfer to "tflite"   
   
#### tflite support operator  
ADD, ARG_MAX, CAST, CONV_2D, FULLY_CONNECTED, LESS, LOGICAL_AND, RANGE, RESHAPE, SOFTMAX, SUB  
#### use operators if using tf.dataset to read image file
DecodeJpeg, Enter, Exit, LoopCond, Merge, ReadFile, Switch, TensorArrayGatherV3, TensorArrayReadV3, TensorArrayScatterV3, TensorArraySizeV3, TensorArrayV3, TensorArrayWriteV3  
  
#### It needs to back original method.  To see the "tf1_mnist.py"  
#### Converter use "tf.lite.TFLiteConverter.from_frozen_graph" instead of "tf.lite.toco_convert"  
  
  
  
  
### Tf2 (tf 2.1.0)  
Solution 1: "pb file" transfer to "tflite"  
The source code is in the "tf2_mnist_pb.py".  
  
It can use subclassing.
```python  
#save pb
tf.saved_model.save(model, "./mnist_tf2/")    #This means "saving pb"
converter = tf.lite.TFLiteConverter.from_saved_model("./mnist_tf2/")
tflite_model = converter.convert()
open('mnist_tf2_savepb.tflite', 'wb').write(tflite_model)
```

