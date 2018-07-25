# import library yang dibutuhkan
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from helpers import *

# load data (apabila pertama kali menjalankan, maka script ini otomatis mendownload data)
print('[INFO] Menyiapkan Data ...')
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

n_classes = 10
batch_size = 128

# inisialisasi placeholder untuk data dan label
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

# dropout
hold_prob = tf.placeholder(tf.float32)

# buat arsitektur jaringan
def convolutional_neural_network(X, y):
	X = tf.reshape(X, shape=[-1, 28, 28, 1])

	conv1 = convolutional_layer(X, shape=[5,5,1,32])
	conv1 = max_pooling_2by2(conv1)

	conv2 = convolutional_layer(conv1, shape=[5,5,32,64])
	conv2 = max_pooling_2by2(conv2)

	fc = tf.reshape(conv2, [-1,7*7*64])
	fc = tf.nn.relu(normal_full_layer(fc, 1024))

	# dropout
	#hold_prob = tf.placeholder(tf.float32)
	fc_dropout = tf.nn.dropout(fc, keep_prob=hold_prob)

	output = normal_full_layer(fc, 10)

	return output

# buat fungsi training neural network
def training(X, y, hm_epochs=10):
	print("[INFO] Melatih Jaringan ...")
	# hasil prediksi = yang dihasilkan dari fungsi neural net
	prediction = convolutional_neural_network(X, y)
	# definisikan cost / loss function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	# tentukan metode optimisasi
	optimizer = tf.train.AdamOptimizer()
	train = optimizer.minimize(cost)

	# jalankan komputasi graf dengan session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			# train per batch
			for _ in range(int(mnist.train.num_examples / batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([train, cost], feed_dict={X:epoch_x, y:epoch_y, hold_prob:0.7})
				epoch_loss += c
			print("Epoch : ",epoch, " / ", hm_epochs," Loss:", epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		print("Akurasi :", accuracy.eval({X:mnist.test.images, y:mnist.test.labels, hold_prob:1.0}))

training(X, y)