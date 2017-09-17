import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Print cross entropy from session



def cross_entropy():
   cross = None
   softmax_data = [0.7, 0.2, 0.1]
   one_hot_data = [1.0, 0.0, 0.0]

   softmax = tf.placeholder(tf.float32)
   one_hot = tf.placeholder(tf.float32)
   loga = tf.placeholder(tf.float32)
   values = tf.placeholder(tf.float32)
   cross = tf.placeholder(tf.float32)
   m = tf.placeholder(tf.float32)

   logaritmo = tf.log(softmax)
   multiplicacion = tf.multiply(one_hot,loga)
   suma = tf.reduce_sum(values)
   crossEntropy = tf.multiply(cross,m)

   
   with tf.Session() as sess:
      a = sess.run(logaritmo,feed_dict={softmax:softmax_data})
      print(a)
      b = sess.run(multiplicacion,feed_dict={one_hot:one_hot_data, loga:a})
      print(b)
      c = sess.run(suma,feed_dict={values:b})
      print(c)
      cross = sess.run(crossEntropy,feed_dict={cross:c,m:-1})
      print(cross)
   return cross

c = cross_entropy()
print (c)

