import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print("[INFO] Load Data ..")
iris = load_iris()
data = iris.data
label = iris.target

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# hyperparameter NN

input_node = 4
hidden_node = 2
output_node = 4

X = tf.placeholder(tf.float32, [None, input_node])

def autoencoder(X):
	tf.set_random_seed(46)
	hidden_layer = {'weights' : tf.Variable(tf.random_normal([input_node, hidden_node])),
			  'biases' : tf.Variable(tf.random_normal([hidden_node]))}

	output_layer = {'weights' : tf.Variable(tf.random_normal([hidden_node, output_node])),
			  'biases' : tf.Variable(tf.random_normal([output_node]))}

	l1 = tf.add(tf.matmul(X, hidden_layer['weights']), hidden_layer['biases'])

	output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])

	return output, l1

def train_neural_network(X):
	prediction, l1 = autoencoder(X)
	cost = tf.reduce_mean(tf.square(prediction - X))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	hm_epochs = 1000

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			_, c = sess.run([optimizer, cost], feed_dict={X:data})

			print('Epoch :', epoch, '/', hm_epochs, 'loss:', c)

		l = sess.run(l1, feed_dict = {X: data})

	return l

l = train_neural_network(X)

plt.scatter(l[:, 0], l[:, 1], c=label, cmap='RdYlBu')
plt.show()
























