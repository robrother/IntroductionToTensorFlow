import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

x = tf.constant(10.0)
y = tf.constant(2.0)
a = tf.constant(1.0)
z = tf.placeholder(tf.float32)

# TODO: Convert the following to TensorFlow:
with tf.Session() as sess:
   z = x/y - a
   output = sess.run(z)
   print (output)
