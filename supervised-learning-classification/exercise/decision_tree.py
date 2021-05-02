import os, types
import pandas as pd

from pathlib import Path

path = Path().absolute()
filePath = path/'Basic Machine Learning/supervised-learning-classification/exercise/Iris.csv'

iris = pd.read_csv(filePath)

# Menghilangkan kolom yang tidak penting
iris.drop('Id',axis=1,inplace=True)

# Memisahkan atribut dan label
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
y = iris['Species']

# print(iris.head())

from sklearn.tree import DecisionTreeClassifier
 
# Membuat model Decision Tree
tree_model = DecisionTreeClassifier() 
 
# Melakukan pelatihan model terhadap data
tree_model.fit(X, y)

# Prediksi model dengan tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
tree_model.predict([[6.2, 3.4, 5.4, 2.3]])

# Melihat visualisasi dari decision tree yang kita buat terhadap data dengan menggunakan library Graphviz
from sklearn.tree import export_graphviz
export_graphviz(
    tree_model,
    out_file = "Basic Machine Learning/supervised-learning-classification/exercise/iris_tree.dot",
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica' ],
    rounded= True,
    filled =True
)

# Mendapatkan output berupa berkas iris_tree.dot
# Pastikan output iris_tree.dot-nya sudah ada dengan memanggil os.listdir.
dotFilePath = path/'Basic Machine Learning/supervised-learning-classification/exercise/'
for file in os.listdir(dotFilePath):
    if file.endswith(".dot"):
        print(os.path.join(dotFilePath, file))