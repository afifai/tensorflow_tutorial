# import library yang dibutuhkan
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data (apabila pertama kali menjalankan, maka script ini otomatis mendownload data)
print('[INFO] Menyiapkan Data ...')
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# hyperparameter
n_nodes_l1 = 500
n_nodes_l2 = 700
n_nodes_l3 = 300

n_classes = 10
batch_size = 128

# inisialisasi placeholder untuk data dan label
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

# buat arsitektur jaringan
def neural_networks_model(X, y):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_l1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_l1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_l1,n_nodes_l2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_l2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_l2,n_nodes_l3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_l3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_l3,n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}


	l1 = tf.add(tf.matmul(X, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	return output

# buat fungsi training neural network
def training(X, y, hm_epochs=10):
	print("[INFO] Melatih Jaringan ...")
	# hasil prediksi = yang dihasilkan dari fungsi neural net
	prediction = neural_networks_model(X, y)
	# definisikan cost / loss function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	# tentukan metode optimisasi
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
	train = optimizer.minimize(cost)

	# jalankan komputasi graf dengan session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			# train per batch
			for _ in range(int(mnist.train.num_examples / batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([train, cost], feed_dict={X:epoch_x, y:epoch_y})
				epoch_loss += c
			print("Epoch : ",epoch, " / ", hm_epochs," Loss:", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		print("Akurasi :", accuracy.eval({X:mnist.test.images, y:mnist.test.labels}))

training(X, y)