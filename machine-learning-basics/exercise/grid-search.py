import pandas as pd

filePath = '/Users/fadhilhanri/Documents/Challenge/Basic Machine Learning/machine-learning-basics/exercise/Salary_Data.csv'

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

# Implemen Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
 
# Membangun model dengan parameter C, gamma, dan kernel
# Parameter pertama (model) adalah model yang akan kita uji
# Parameter kedua (parameters) adalah dictionary yang berisi kumpulan parameter dari model yang akan diuji
model = SVR()
parameters = {
    'kernel': ['rbf'],
    'C':     [1000, 10000, 100000],
    'gamma': [0.5, 0.05,0.005]
}
grid_search = GridSearchCV(model, parameters)
 
# Melatih model dengan fungsi fit
grid_search.fit(X,y)

# Menampilkan parameter terbaik dari objek grid_search
# print(grid_search.best_params_)

# Membuat model SVM baru dengan parameter terbaik hasil grid search
model_baru  = SVR(C=100000, gamma=0.005, kernel='rbf')
model_baru.fit(X,y)

# Memvisualisasikan model
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, model_baru.predict(X))
plt.show()