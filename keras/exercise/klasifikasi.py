import tensorflow as tf
# Cek versi tensorflow, instal versi terbaru dengan: pip install tensorflow
# print(tf.__version__)

import os

# Pada direktori data latih dan data validasi masing-masing memiliki sub-direktori clean dan messy
# Setiap sub-direktori menyimpan gambar yang sesuai dengan nama sub-direktori tersebut
# Sub-direktori ‘clean’ terdapat gambar-gambar ruangan yang rapi dan pada sub-direktori ‘messy’ terdapat gambar-gambar ruangan yang berantakan
train_dir = '/Users/fadhilhanri/Documents/Challenge/Basic Machine Learning/keras/exercise/images/train'
validation_dir = '/Users/fadhilhanri/Documents/Challenge/Basic Machine Learning/keras/exercise/images/val'
# print(os.listdir(train_dir))
# print(os.listdir(validation_dir))

# Tampung direktori dari setiap kelas pada direktori latih dan direktori validasi ke dalam variabel
# Pembuatan direktori di sini akan dipakai saat menggunakan objek image data generator

# Membuat direktori ruangan rapi pada direktori data training
train_clean_dir = os.path.join(train_dir, 'clean')
# print(os.listdir(train_clean_dir))
 
# Membuat direktori ruangan berantakan pada direktori data training
train_messy_dir = os.path.join(train_dir, 'messy')
# print(os.listdir(train_messy_dir))
 
# Membuat direktori ruangan rapi pada direktori data validasi
validation_clean_dir = os.path.join(validation_dir, 'clean')
# print(os.listdir(validation_clean_dir))
 
# Membuat direktori ruangan berantakan pada direktori data validasi
validation_messy_dir = os.path.join(validation_dir, 'messy')
# print(os.listdir(validation_messy_dir))

# Buat sebuah objek ImageDataGenerator untuk data training dan data testing
# Image data generator adalah sebuah fungsi yang sangat berguna untuk mempersiapkan data latih dan data testing yang akan diberikan ke model
# Beberapa kemudahan yang disediakan Image data generator adalah, preprocessing data, pelabelan sampel otomatis, dan augmentasi gambar

# Augmentasi gambar adalah teknik untuk menciptakan data-data baru dari data yang telah ada
# Contoh augmentasi gambar adalah horizontal flip di mana gambar akan dibalikkan secara horizontal
# Augmentasi gambar: https://keras.io/preprocessing/image/

from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    rotation_range = 20,
                    horizontal_flip = True,
                    shear_range = 0.2,
                    fill_mode = 'nearest')
 
test_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    rotation_range = 20,
                    horizontal_flip = True,
                    shear_range = 0.2,
                    fill_mode = 'nearest')

# Lalu kita dapat menggunakan objek image data generator sebelumnya untuk mempersiapkan data latih yang akan dipelajari oleh model

train_generator = train_datagen.flow_from_directory(
        train_dir,              # direktori data latih
        target_size=(150, 150), # mengubah resolusi seluruh gambar menjadi 150x150 piksel
        batch_size=4, 
        class_mode='binary')    # karena kita merupakan masalah klasifikasi 2 kelas maka menggunakan class_mode = 'binary'
 
validation_generator = test_datagen.flow_from_directory(
        validation_dir,         # direktori data validasi
        target_size=(150, 150), # mengubah resolusi seluruh gambar menjadi 150x150 piksel
        batch_size=4,           
        class_mode='binary')    # karena kita merupakan masalah klasifikasi 2 kelas maka menggunakan class_mode = 'binary'

# Setelah data telah siap, kita bisa membangun arsitektur sebuah CNN
# Sebuah CNN pada keras mirip dengan MLP untuk klasifikasi fashion MNIST
# Perbedaannya hanya pada terdapatnya 2 lapis layer konvolusi dan max pooling
# Fungsi dari layer konvolusi adalah untuk mengekstraksi atribut pada gambar
# Sedangkan layer max pooling berguna untuk mereduksi resolusi gambar sehingga proses pelatihan MLP lebih cepat

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Setelah membuat arsitektur dari CNN, jangan lupa untuk memanggil fungsi compile pada objek model, dan tentukan loss function serta optimizer

# Compile model dengan 'adam' optimizer loss function 'binary_crossentropy'
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# Setelah menentukan loss function dan optimizer pada CNN, kita dapat melatih model kita menggunakan metode fit
# Dengan menggunakan image data generator, kita tidak perlu memasukkan parameter gambar dan labelnya
# Image data generator secara otomatis melabeli sebuah gambar sesuai dengan direktori di mana ia disimpan
# Contohnya sebuah gambar yang terdapat di direktori clean, secara otomatis akan diberi label “clean” oleh image data generator

# latih model dengan model.fit 
model.fit(
      train_generator,
      steps_per_epoch=25,                   # berapa batch yang akan dieksekusi pada setiap epoch
      epochs=20,                            # tambahkan eposchs jika akurasi model belum optimal
      validation_data=validation_generator, # menampilkan akurasi pengujian data validasi
      validation_steps=5,                   # berapa batch yang akan dieksekusi pada setiap epoch
      verbose=2)

