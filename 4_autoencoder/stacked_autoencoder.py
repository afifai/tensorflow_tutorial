import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('data_MNIST/', one_hot=True)

input_node = 784
n_nodes_h1 = 621
n_nodes_h2 = 312
n_nodes_h3 = 128
n_nodes_h4 = 312
n_nodes_h5 = 621
n_classes = 784
batch_size = 128

scaler = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

# noise matrix
mean = 0.9
stddev = 0.7
noise_global = np.random.normal(mean, stddev, 784)


def stacked_an(X):
	hidden_1_layer = {'weights' : tf.Variable(scaler([input_node, n_nodes_h1], dtype=tf.float32)),
			  'biases' : tf.Variable(scaler([n_nodes_h1], dtype=tf.float32))}

	hidden_2_layer = {'weights' : tf.Variable(scaler([n_nodes_h1,n_nodes_h2], dtype=tf.float32)),
			  'biases' : tf.Variable(scaler([n_nodes_h2], dtype=tf.float32))}
	
	hidden_3_layer = {'weights' : tf.Variable(scaler([n_nodes_h2, n_nodes_h3], dtype=tf.float32)),
			  'biases' : tf.Variable(scaler([n_nodes_h3], dtype=tf.float32))}

	hidden_4_layer = {'weights' : tf.Variable(scaler([n_nodes_h3, n_nodes_h4], dtype=tf.float32)),
			  'biases' : tf.Variable(scaler([n_nodes_h4], dtype=tf.float32))}

	hidden_5_layer = {'weights' : tf.Variable(scaler([n_nodes_h4, n_nodes_h5], dtype=tf.float32)),
			  'biases' : tf.Variable(scaler([n_nodes_h5], dtype=tf.float32))}

	output_layer = {'weights' : tf.Variable(scaler([n_nodes_h5, n_classes], dtype=tf.float32)),
			  'biases' : tf.Variable(scaler([n_classes], dtype=tf.float32))}

	l1 = tf.add(tf.matmul(X, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
	l4 = tf.nn.relu(l4)

	l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
	l5 = tf.nn.relu(l5)

	output = tf.add(tf.matmul(l5, output_layer['weights']), output_layer['biases'])
	output = tf.nn.relu(output)

	return output

def train_neural_network(X,y):
	prediction = stacked_an(X)
	cost = tf.reduce_mean(tf.square(prediction-y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
	
		for epoch in range(hm_epochs):
			for _ in range(int(mnist.train.num_examples / batch_size)):
				noise = np.random.normal(mean, stddev, 784)
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, loss = sess.run([optimizer, cost], feed_dict={X: epoch_x + noise, y:epoch_x})
			#loss = cost.eval({X:epoch_x + noise, y:epoch_x})
			print("Epoch : ", epoch, "/", hm_epochs, "Loss :", loss)

		res = prediction.eval({X:mnist.test.images[10:20] + noise_global})

	return res

res = train_neural_network(X,y)


# plot
f, a = plt.subplots(3, 10, figsize=(20,4))
for i in range(10):
	a[0][i].imshow(np.reshape(mnist.test.images[i+10], (28, 28)))
	a[1][i].imshow(np.reshape(mnist.test.images[i+10] + noise_global, (28, 28)))
	a[2][i].imshow(np.reshape(res[i], (28, 28)))
plt.show()
























