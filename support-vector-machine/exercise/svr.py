import pandas as pd

filePath = '/Users/fadhilhanri/Documents/Challenge/Basic Machine Learning/support-vector-machine/exercise/Salary_Data.csv'

df = pd.read_csv(filePath)
# print(df.head())

# Cek apakah terdapat nilai-nilai yang hilang pada dataset serta apakah ada atribut yang bukan berisi bilangan numerik
# df.info()

# Memisahkan atribut dan label
X = df['YearsExperience']
y = df['Salary']
 
# Ketika hanya terdapat satu atribut pada dataframe, maka atribut tersebut perlu diubah bentuknya agar bisa diterima oleh model dari library SKLearn
# Untuk mengubah bentuk atribut kita membutuhkan library numpy
import numpy as np

# Support for multi-dimensional indexing (e.g. `obj[:, None]`) is deprecated and will be removed in a future version.  
# Convert to a numpy array before indexing instead.
X = X[:,np.newaxis]

# Buat objek support vector regression dan di sini kita akan mencoba menggunakan parameter C = 1000, gamma = 0.05, dan kernel ‘rbf’
from sklearn.svm import SVR
 
# Membangun model dengan parameter C, gamma, dan kernel
model  = SVR(C=1000, gamma=0.05, kernel='rbf')
 
# Melatih model dengan fungsi fit
model.fit(X,y)

# Visualisasikan bagaimana model SVR kita menyesuaikan terhadap pola yang terdapat pada data menggunakan library matplotlib
import matplotlib.pyplot as plt
 
# Memvisualisasikan model
plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.show()