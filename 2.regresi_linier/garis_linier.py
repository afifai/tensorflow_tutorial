# import library yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt

# buat titik sepanjang sumbu y=x 
# dengan jarak 0.1 dari titik x=0 sampai x=5
X = np.arange(0.0, 5.0, 0.1)
print("Data X :")
print(X)

# definisikan a & b
a = 1
b = 0
# bentuk garis linier Y = aX + b
Y = a*X + b
# gambar plotnya
plt.plot(X,Y)
plt.ylabel("Variabel Terikat")
plt.xlabel("Variabel Bebas")
plt.show()
