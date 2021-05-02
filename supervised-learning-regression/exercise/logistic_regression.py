import pandas as pd

from pathlib import Path

# path = Path().absolute()
filePath = '/Users/fadhilhanri/Documents/Challenge/Basic Machine Learning/unsupervised-learning-clustering/exercise/Social_Network_Ads.csv'

df = pd.read_csv(filePath)
# print(df.head())

# Kita dapat melihat apakah ada nilai yang kosong pada setiap atribut dengan menggunakan fungsi info()
# print(df.info())

# Pada dataset terdapat kolom ‘User ID’
# Kolom tersebut merupakan atribut yang tidak penting untuk dipelajari oleh model sehingga perlu dihilangkan
# Untuk menghilangkan kolom dari dataframe, gunakan fungsi drop

data = df.drop(columns=['User ID'])

# Jalankan proses one-hot encoding dengan pd.get_dummies()
data = pd.get_dummies(data)
# print(data)

# Pisahkan antara atribut dan label
predictions = ['Age' , 'EstimatedSalary' , 'Gender_Female' , 'Gender_Male']
X = data[predictions]
y = data['Purchased']

# Lakukan normalisasi terhadap data yang kita miliki
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_data = pd.DataFrame(scaled_data, columns= X.columns)
scaled_data.head()
# print(scaled_data.head())

# Bagi data menjadi train dan test set dengan fungsi train_test_split yang disediakan SKLearn
from sklearn.model_selection import train_test_split
 
# Bagi data menjadi train dan test untuk setiap atribut dan label
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=1)

# Buat model dengan membuat sebuah objek logistic regression
from sklearn import linear_model
 
# Latih model dengan fungsi fit
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

# Uji akurasi model pada test set dengan memanggil fungsi score() pada objek model
from sklearn.preprocessing import StandardScaler

model.score(X_test, y_test)
# print(model.score(X_test, y_test))