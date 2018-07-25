import tensorflow as tf

# inisialisasi bobot
def init_weights(shape):
	init_random_dist = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init_random_dist)

# inisialisasi bias
def init_bias(shape):
	init_bias_vals = tf.constant(0.1, shape=shape)
	return tf.Variable(init_bias_vals)

# CONV 2D
def conv2d(x, W):
	# x -- > [batch, H, W, C]
	# W -- > [filter H, filter W, channel_IN, channel_OUT]
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# POOLING 2D
def max_pooling_2by2(x):
	# x --> [batch, H, W, C]
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_layer(input_x, shape):
	W = init_weights(shape)
	b = init_bias([shape[-1]])
	return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
	input_size = int(input_layer.get_shape()[1])
	W = init_weights([input_size, size])
	b = init_bias([size])
	return tf.add(tf.matmul(input_layer, W), b)
