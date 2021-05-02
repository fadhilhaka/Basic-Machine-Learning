import pandas as pd

from pathlib import Path

# path = Path().absolute()
filePath = '/Users/fadhilhanri/Documents/Challenge/Basic Machine Learning/unsupervised-learning-clustering/exercise/Mall_Customers.csv'

# Ubah file csv menjadi dataframe
df = pd.read_csv(filePath)
 
# Tampilkan 3 baris pertama
# print(df.head(3))

# Ubah nama kolom
df = df.rename(columns={'Gender': 'gender', 'Age': 'age',
                        'Annual Income (k$)': 'annual_income',
                        'Spending Score (1-100)': 'spending_score'})
 
# Ubah data kategorik menjadi data numerik
df['gender'].replace(['Female', 'Male'], [0,1], inplace=True)
 
# Tampilkan data yang sudah di preprocess
# print(df.head(3))

from sklearn.cluster import KMeans
 
# Menghilangkan kolom customer id dan gender
X = df.drop(['CustomerID', 'gender'], axis=1)
 
# Membuat list yang berisi inertia
clusters = []
for i in range(1,11):
  km = KMeans(n_clusters=i).fit(X)
  clusters.append(km.inertia_)

# Membuat plot inertia
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Cari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')
# plt.show()

# Membuat objek KMeans
km5 = KMeans(n_clusters=5).fit(X)
 
# Menambahkan kolom label pada dataset
X['Labels'] = km5.labels_
 
# Membuat plot KMeans dengan 5 klaster
plt.figure(figsize=(8,4))
sns.scatterplot(X['annual_income'], X['spending_score'], hue=X['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('KMeans dengan 5 Cluster')
plt.show()