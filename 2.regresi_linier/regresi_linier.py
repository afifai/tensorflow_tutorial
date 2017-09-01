# import library yang dibutuhkan
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# TAHAP PERSIAPAN
# buat data (toy dataset)
x_data = np.random.rand(100).astype(np.float32)

# bentuk data akan disesuaikan
# dengan persamaan Y = 2X + 3
y_data = x_data * 3 + 3
y_data = np.vectorize(lambda y: y+np.random.normal(loc=0.0, scale=0.1))(y_data)

# tampilkan sampel data
print "Contoh bentuk data:"
print(zip(x_data, y_data)[0:5])

# PROSES REGRESI MENGGUNAKAN TENSORFLOW
# inisialisasi variabel a dan b (weight)
# diikuti dengan fungsi linier y = a*x + b
a = tf.Variable(0.5)
b = tf.Variable(0.2)
y = a * x_data + b

# inisialisasi fungsi loss (Mean Square Error)
loss = tf.reduce_mean(tf.square(y - y_data))

# inisialisasi optimizer (gradient descent) dengan learning rate=0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# inisialisasi semua variabel sebelum inisialisasi session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# lakukan pelatihan terhadap data, dan simpan data per 5 epoch
train_data = []
for step in range(100):
	evals = sess.run([train, a, b])[1:]
	if step % 5 == 0:
		print(step, evals)
		train_data.append(evals)

sess.close()
# PROSES PLOT
#converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
	cb += 1.0 / len(train_data)
	cg -= 1.0 / len(train_data)
	if cb > 1.0: cb = 1.0
	if cg < 0.0: cg = 0.0
	[a, b] = f
	f_y = np.vectorize(lambda x: a*x + b)(x_data)
	line = plt.plot(x_data, f_y)
	plt.setp(line, color=(cr, cg, cb))

plt.plot(x_data, y_data, 'ro')
plt.xlabel("Variabel Bebas")
plt.ylabel("Variabel Terikat")
plt.show()
