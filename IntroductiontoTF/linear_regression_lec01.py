import numpy as np 
##import seaborn 
import matplotlib.pyplot as plt 
import tensorflow as tf


# Define the input data 
X_data = np.arange(100, step = 0.1)
y_data = X_data + 20 * np.sin(X_data/10)


## Visualize the input data 
plt.scatter(X_data, y_data, color = 'blue')



## Defining the model of for linear regression in tensorflow 
n_samples = 1000 
batch_size = 100 

## Resizing the input data, just to be safe 
X_data = np.reshape(X_data, (n_samples,1))
y_data = np.reshape(y_data, (n_samples,1))


## Defing the placeholder variables to feed the data in 
X = tf.placeholder(tf.float32, shape = (batch_size,1))
y = tf.placeholder(tf.float32, shape = (batch_size,1))

#print("Declared variabels")

# Define variables to be learned 
with tf.variable_scope("linear_regression"):
	W = tf.get_variable("weights", (1,1), initializer = tf.random_normal_initializer())
	b = tf.get_variable("bias", (1,), initializer = tf.constant_initializer(0.0))

	y_pred = tf.matmul(X,W) + b 
	loss = tf.reduce_sum((y-y_pred)**2/n_samples)

opt_operation = tf.train.AdamOptimizer().minimize(loss)

print("Reached here....")

## Is there a way to run this in parallel
with tf.Session() as sess: 
	#Initialize variables in graph
	sess.run(tf.initialize_all_variables())
	# Gradient descent loop for minibatches
	for _ in range(500):
		indices = np.random.choice(n_samples, batch_size)
		X_batch, y_batch = X_data[indices], y_data[indices]
		_, loss_val = sess.run([opt_operation, loss], feed_dict = {X : X_batch, y: y_batch})
		print(loss_val)

	print("printing the parameters")
	print(sess.run(W))
	print(sess.run(b))
	yP = sess.run(X_data*W + b) 
	## Sanity check on the shape of the parameters coming out of the graph
	print(yP.shape)
	print(X_data.shape)
	plt.scatter(X_data, yP, color = 'red')
	plt.show()




## Visualizing the result and the learned weights and biases