# import library tensorflow
import tensorflow as tf

# deklarasikan dua konstanta
a = tf.constant([5])
b = tf.constant([10])

# lakukan operasi penjumlahan
c = tf.add(a, b) # operasi menggunakan tensorflow
# c = a+b # operasi menggunakan operasi dasar python

# Inisialisasi session
# cara pertama
session = tf.Session()

# jalankan session
hasil = session.run(c)
print("Menggunakan session cara 1")
print(hasil)

# tutup session untuk merelease resource komputasi yang digunakan
session.close()

# cara 2
with tf.Session() as session:
	hasil = session.run(c)
	print("Menggunakan session cara 2")
	print(hasil)


