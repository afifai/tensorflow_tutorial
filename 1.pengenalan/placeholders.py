# import library tensorflow
import tensorflow as tf

# deklarasikan placeholder a dengan tipe float 32 bit
a = tf.placeholder(tf.float32)

# definisikan sebuah fungsi
b = a*2

# deklarasikan session
with tf.Session() as sess:
	hasil = sess.run(b, feed_dict={a:2.5})
	print(hasil)

# buat data yang dimasukkan berupa array
dictionary = {a: [[1, 2, 3],[4, 5, 6],[7, 8, 9]]}
with tf.Session() as sess:
        hasil = sess.run(b, feed_dict=dictionary)
        print(hasil)

