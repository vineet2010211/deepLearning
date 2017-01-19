import tensorflow as tf 

x = tf.constant(3.0, name = 'x')
y = tf.constant(2.0, name = 'y')
z = tf.constant(5.0, name = 'z')


op1 = tf.add(y,z)
op2 = tf.mul(x,op1)

with tf.Session() as sess:
	result = sess.run([op2,op1]) ## Fetching the data out of the model
	print(result)



## Constants: baises , Variables model parameters which are usually updated in the network
## , placeholders (feed in the data in the network) feed_dict = {x:[],y:[]}