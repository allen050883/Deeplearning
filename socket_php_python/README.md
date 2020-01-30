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
