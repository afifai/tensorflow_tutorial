# Import library TensorFlow
import tensorflow as tf

# pendefinisian skalar
skalar = tf.constant([5])
# pendefinisian vektor (2D)
vektor = tf.constant([1,2,3])
# pendefinisian matriks (3D)
matriks = tf.constant([[1,2,3],[4,5,6]])
# pendefinisian tensor (4D)
tensor = tf.constant([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])

# buat session 
with tf.Session() as session:
	hasil = session.run(skalar)
	print "Hasil skalar : \n %s \n" %hasil
        hasil = session.run(vektor)
        print "Hasil vektor : \n %s \n" %hasil
        hasil = session.run(matriks)
        print "Hasil matriks : \n %s \n" %hasil
        hasil = session.run(tensor)
        print "Hasil tensor : \n %s \n" %hasil

