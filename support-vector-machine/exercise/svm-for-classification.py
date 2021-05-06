import pandas as pd

filePath = '/Users/fadhilhanri/Documents/Challenge/Basic Machine Learning/support-vector-machine/exercise/diabetes.csv'

df = pd.read_csv(filePath)
# print(df.head())

# Cek apakah terdapat nilai-nilai yang hilang pada dataset serta apakah ada atribut yang bukan berisi bilangan numerik
# df.info()

# Memisahkan atribut pada dataset dan menyimpannya pada sebuah variabel
X = df[df.columns[:8]]
# print(X)
 
# Memisahkan label pada dataset dan menyimpannya pada sebuah variabel
y = df['Outcome']
# print(y)

# Ubah nilai-nilai dari setiap atribut agar berada pada skala yang sama
from sklearn.preprocessing import StandardScaler
 
# Standarisasi nilai-nilai dari dataset
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# print(X)

# Pisah data untuk training dan testing menggunakan fungsi .train_test_split()
from sklearn.model_selection import train_test_split
 
# Memisahkan data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Buat objek Support Vector Classifier dan tampung dalam variabel clf
from sklearn.svm import SVC
 
# Membuat objek SVC dan memanggil fungsi fit untuk melatih model
clf = SVC()
clf.fit(X_train, y_train)

# Menampilkan skor akurasi prediksi
score = clf.score(X_test, y_test)
print(score)