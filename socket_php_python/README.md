# socket communication
This shows the example for php client sending mnist file to php server.  

## php client  
```php
<?php
//set localhost IP and give a port number
$host    = "127.0.0.1";
$port    = 54321;

//try to scan all files in the directory
$filepath = 'C:\\Users\\user\\Desktop\\php_test\\mnist_test\\';
$path_parts = scandir($filepath, 1);

//make all array in string in order to send msg (array is not the good choice)
$all_file =  implode("\n", $path_parts);

//connect the socket and wait for server
$socket = socket_create(AF_INET, SOCK_STREAM, 0) or die("Could not create socket\n");
$result = socket_connect($socket, $host, $port) or die("Could not connect to server\n");  
socket_write($socket, $all_file, strlen($all_file)) or die("Could not send data to server\n");
$result = socket_read ($socket, 4096) or die("Could not read server response\n");
echo "The answer From Server :".$result;
socket_close($socket);
?>
```
  
## python server  
```python
import time
import socket
import tensorflow as tf

filepath = 'C:\\Users\\user\\Desktop\\php_test\\mnist_test\\';

#build the socket accept
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('localhost', 54321))
sock.listen(1)
(csock, adr) = sock.accept()

#always run model
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open("mnist.pb", "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name = "")
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input_x = sess.graph.get_tensor_by_name("input:0")
        softmax = sess.graph.get_tensor_by_name("result/softmax:0")
        pred = sess.graph.get_tensor_by_name("result/prediction:0")
        
        while True:
            #try to keep the connection from client, if lose th connection and retry to connect
            try:
                msg = csock.recv(4096)
                msg_decode = msg.decode()
                #print("Client send: " + msg_decode)
                msg_split = msg_decode.split("\n")
                msg_list = []
                for m in msg_split:
                    if '.jpg' in m:
                        msg_list.append(filepath + m)
                
                pred_num, soft = sess.run([pred, softmax], feed_dict={input_x: msg_list})
                all_answer = " ".join(str(x) for x in pred_num)
                print(all_answer)
                msg_recall = "Hello I'm Server. The num is " + all_answer
                msg_encode = msg_recall.encode()
                csock.send(msg_encode)
                csock.close()
                
            except:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('localhost', 54321))
                sock.listen(1)
                start_time = time.time()
                (csock, adr) = sock.accept()
                print(time.time() - start_time)

```
