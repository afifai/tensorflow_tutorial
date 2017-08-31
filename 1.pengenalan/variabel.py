# import library tensorflow
import tensorflow as tf

# definisikan variabel
tes = tf.Variable([1])

# buat komputasi graf counter sederhana
kons = tf.constant([1])
nilai_baru = tf.add(tes, kons)
update = tf.assign(tes, nilai_baru)

# inisialisasi variabel
init_op = tf.global_variables_initializer()

# deklarasikan session
with tf.Session() as sess:
	sess.run(init_op)
	print(sess.run(tes))
	for _ in range(3):
		sess.run(update)
		print(sess.run(tes))
