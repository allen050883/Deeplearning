# Tensorflow Lite  
### Tf1 (tf 1.14)  
Solution: "pb file" transfer to "tflite"   
  
Notice!!!!!  
#### tflite support operator  
ADD, ARG_MAX, CAST, CONV_2D, FULLY_CONNECTED, LESS, LOGICAL_AND, RANGE, RESHAPE, SOFTMAX, SUB  
#### use operators if using tf.dataset to read image file
DecodeJpeg, Enter, Exit, LoopCond, Merge, ReadFile, Switch, TensorArrayGatherV3, TensorArrayReadV3, TensorArrayScatterV3, TensorArraySizeV3, TensorArrayV3, TensorArrayWriteV3  
  
It needs to back original method.  To see the "tf1_mnist.py"  
Converter use "tf.lite.TFLiteConverter.from_frozen_graph" instead of "tf.lite.toco_convert"  
  
  
### Tf2 (tf 2.1.0)  
