# 
# This is a very simple example to show how to initiate a tf.summary.FileWriter
# that you can launch tensorboard
#

import tensorflow as tf
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_log_dir = "tf_logs"
log_dir = "{}/run-{}/".format(root_log_dir, now)

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# After the graph is defined, pass it to the tf.summary.FileWriter to initialize the FileWriter class
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

with tf.Session() as sess:
    print(sess.run(c))

file_writer.close()

# Once the program is finished, you can fire up the TensorBoard by running tensorboard --logdir tf_logs


