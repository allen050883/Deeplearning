import tensorflow as tf

worker1 = "10.0.0.1:10000"
worker2 = "10.0.0.2:10000"

worker_hosts = [worker1, worker2]

cluster_spec = tf.train.ClusterSpec({ "worker": worker_hosts})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)

server.join()
