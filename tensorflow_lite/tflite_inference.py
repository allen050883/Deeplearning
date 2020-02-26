import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mnist.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

input_data = plt.imread('C:/Users/user/Desktop/php_test/mnist_test/0.0.jpg')
input_data = np.reshape(input_data, (1, 28, 28, 1))
input_data = input_data.astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
