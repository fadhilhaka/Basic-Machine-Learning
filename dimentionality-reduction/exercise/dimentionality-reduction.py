from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
atribut = iris.data
label = iris.target
 
# Bagi dataset menjadi train set dan test set
X_train, X_test, y_train, y_test = train_test_split(atribut, label, test_size=0.2)

# Gunakan model Decision Tree dan menghitung berapa akurasinya tanpa menggunakan PCA
# Akurasi tanpa PCA adalah 0.933
from sklearn import tree
 
decision_tree = tree.DecisionTreeClassifier()
model_pertama = decision_tree.fit(X_train, y_train)
model_pertama.score(X_test, y_test)
# print(model_pertama.score(X_test, y_test))

# Gunakan PCA dan menghitung variance dari setiap atribut
from sklearn.decomposition import PCA
 
# Membuat objek PCA dengan 4 principal component
pca = PCA(n_components=4)
 
# Mengaplikasikan PCA pada dataset
pca_attributes = pca.fit_transform(X_train)
 
# Melihat variance dari setiap atribut
pca.explained_variance_ratio_
# print(pca.explained_variance_ratio_)

# Melihat dari variance sebelumnya kita bisa mengambil 2 principal component terbaik karena total variance nya adalah 0.977 yang sudah cukup tinggi
# PCA dengan 2 principal component
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

# Uji akurasi classifier
model2 = decision_tree.fit(X_train_pca, y_train)
model2.score(X_test_pca, y_test)

# Akurasi dengan PCA, menggunakan 2 principal component  atau 2 atribut, adalah 0.966
print(model2.score(X_test_pca, y_test))