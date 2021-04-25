# Pengolahan Data

## Data Preparation dengan Teknik One-Hot-Encoding

Setelah dataset dibersihkan, masih ada beberapa tahap yang perlu dilakukan agar dataset benar-benar siap untuk diproses oleh model machine learning. Biasanya, dataset Anda akan terdiri dari dua jenis data: kategorik dan numerik.

Contoh data numerik adalah: ukuran panjang, suhu, nilai uang, hitungan dalam bentuk angka, dll, yang terdiri dari bilangan integer (seperti -1, 0, 1, 2, 3, dan seterusnya) atau bilangan float (seperti -1.0, 2.5, 39.99, dan seterusnya).

Setiap nilai dari data dapat diasumsikan memiliki hubungan dengan data lain karena data numerik dapat dibandingkan dan memiliki ukuran yang jelas. Misal, Anda dapat mengatakan bahwa panjang 39 m lebih besar dibanding 21 m. Jenis data ini terdefinisi dengan baik, dapat dioperasikan dengan metode statistik, dan mudah dipahami oleh komputer.

Jenis data lain yang sering kita temui adalah data kategorik. Data kategorik adalah data yang berupa kategori dan berjenis string, tidak dapat diukur atau didefinisikan dengan angka atau bilangan. 

Contoh data kategorik adalah sebuah kolom pada dataset yang berisi perkiraan cuaca seperti cerah, berawan, hujan, atau berkabut. Contoh lain dari data kategorik adalah jenis buah misalnya apel, pisang, semangka, dan jeruk. 

Pada jenis data ini, kita tidak bisa mendefinisikan operasi perbandingan seperti lebih besar dari, sama dengan, dan lebih kecil dari. Dan dengan demikian, kita juga tidak dapat mengurutkan dan melakukan operasi statistik terhadap data jenis ini.

Umumnya, model machine learning tidak dapat mengolah data kategorik, sehingga kita perlu melakukan konversi data kategorik menjadi data numerik. Banyak model machine learning seperti Regresi Linear dan Support Vector Machine (kedua model ini akan dibahas pada modul-modul selanjutnya) yang hanya menerima input numerik sehingga tidak bisa memproses data kategorik. 

Salah satu teknik untuk mengubah data kategorik menjadi data numerik adalah dengan menggunakan **One Hot Encoding** atau yang juga dikenal sebagai **dummy variables**. One Hot Encoding mengubah data kategorik dengan membuat kolom baru untuk setiap kategori seperti gambar di bawah.

![](https://lh4.googleusercontent.com/zOUhYcTGEbw0DJHjmpStkC2ShR38lR8ZZY676cr9xspCoxkXGWogYyQgovc5YCe5qRqRR14L2L1-kL3e6EzvpqLEBeG1A_Dg2YiNuBS-TpZiXh3TsC9yNVpIbPVdvV1pVmPChuXL)

## Data Preparation dengan Normalization dan Standardization

### Outlier Removal

Dalam statistik, outlier adalah sebuah nilai yang jauh berbeda dari kumpulan nilai lainnya dan dapat mengacaukan hasil dari sebuah analisis statistik. Outlier dapat disebabkan oleh kesalahan dalam pengumpulan data atau nilai tersebut benar ada dan memang unik dari kumpulan nilai lainnya.

Apa pun alasan kemunculannya, Anda perlu tahu cara mengidentifikasi dan memproses outlier. Ini adalah bagian penting dalam persiapan data di dalam machine learning. Salah satu cara termudah untuk mengecek apakah terdapat outlier dalam data kita adalah dengan melakukan visualisasi.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20201223164503f26a8f947fc595252a8e8d464cf69a54.png)

### Normalization

Normalization adalah salah satu teknik yang dipakai dalam data preparation. Tujuan dari normalisasi adalah mengubah nilai-nilai dari sebuah fitur ke dalam skala yang sama. Normalization memungkinkan kenaikan performa dan stabilitas dari sebuah model machine learning.

| Nama | Gaji | Umur | 
|------|------|------|
| A | 12.000.000 | 33 | 
| B | 35.000.000 | 45 | 
| C | 4.000.000 | 23 | 
| D | 6.500.000 | 26 | 
| E | 9.000.000 | 29 |

Contoh dari normalization adalah ketika kita memiliki dataset seperti di atas yang memiliki fitur umur dengan skala 23 sampai 45 tahun dan fitur penghasilan dengan skala 4.000.000 sampai 35.000.000. Di sini kita melihat bahwa fitur penghasilan sekitar satu juta kali lebih besar dari fitur umur dan menunjukkan kedua fitur ini berada pada skala yang sangat jauh berbeda.

Ketika membangun model seperti regresi linear, fitur penghasilan akan sangat mempengaruhi prediksi dari model karena nilainya yang jauh lebih besar daripada umur, walaupun tidak berarti fitur tersebut jauh lebih penting dari fitur umur.

Salah satu contoh dari normalization adalah min-max scaling di mana nilai-nilai dipetakan ke dalam skala 0 sampai 1. SKLearn menyediakan library untuk normalization

Pada Colab kita Import library MinMaxScaler dan masukkan data dari tabel sebelumnya.

~~~
from sklearn.preprocessing import MinMaxScaler
data = [[12000000, 33], [35000000, 45], [4000000, 23], [6500000, 26], [9000000, 29]]
~~~

Pada cell selanjutnya kita buat sebuah objek MinMaxScaler dan panggil fungsi fit() dan mengisi argumen data seperti potongan kode di bawah. Fungsi fit() dari objek MinMaxSclaer adalah fungsi untuk menghitung nilai minimum dan maksimum pada tiap kolom.

~~~
scaler = MinMaxScaler()
scaler.fit(data)
~~~

Sampai pada fungsi fit() ini, komputer baru menghitung nilai minimum dan maksimum pada tiap kolom dan belum melakukan operasi scaler pada data. Terakhir kita panggil fungsi transform() yang akan mengaplikasikan scaler pada data, sebagai berikut.

~~~
print(scaler.transform(data))
~~~

Setiap nilai dari kolom gaji dan umur telah dipetakan pada skala yang sama seperti di bawah ini.

| Nama | Gaji | Umur |
|------|------|------| 
| A | 0.25806452 | 0.45454545 | 
| B | 1 | 1 | 
| C | 0 | 0 | 
| D | 0.08064516 | 0.13636364 |
| E | 0.16129032 | 0.27272727 |

Untuk informasi lebih detail tentang Min Max Scaler, silakan kunjungi [tautan](https://scikit-learn.org/0.16/modules/generated/sklearn.preprocessing.MinMaxScaler.html) berikut.

### Standardization

Standardization adalah proses konversi nilai-nilai dari suatu fitur sehingga nilai-nilai tersebut memiliki skala yang sama. **Z score** adalah metode paling populer untuk standardisasi di mana setiap nilai pada sebuah atribut numerik akan dikurangi dengan rata-rata dan dibagi dengan standar deviasi dari seluruh nilai pada sebuah kolom atribut.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/202012231645430c3498f0a9b96b8b1e7e83a8998784a6.png)

Fungsi standardisasi itu serupa dengan normalization. Keduanya berfungsi menyamakan skala nilai dari tiap atribut pada data. SKLearn menyediakan library untuk mengaplikasikan standard scaler pada data.

Pada colab di cell pertama kita akan mengimpor library preprocessing dari scikit learn lalu membuat data dummy sesuai dengan tabel sebelumnya.

~~~
from sklearn import preprocessing
data = [[12000000, 33], [35000000, 45], [4000000, 23], [6500000, 26], [9000000, 29]]
~~~

Selanjutnya kita buat object scaler dan panggil fungsi fit dari scaler pada data. Fungsi fit memiliki fungsi untuk menghitung rata-rata dan deviasi standar dari setiap kolom atribut untuk kemudian dipakai pada fungsi transform.

~~~
scaler = preprocessing.StandardScaler().fit(data)
~~~

Terakhir, kita panggil fungsi transform untuk mengaplikasikan standard scaler pada data. Untuk melihat hasil dari standard scaler kita tinggal memanggil objek scaler yang telah kita buat sebelumnya. Kodenya sebagai berikut.

~~~
data = scaler.transform(data)
data
~~~

Untuk informasi lebih detail tentang standardization, silakan kunjungi [tautan](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) berikut.

## Data Storage/Warehouse 

Data warehouse pertama kali muncul pada tahun akhir 1980-an. Tujuan awalnya adalah untuk membantu proses aliran data dari sistem operasional ke dalam sistem pendukung keputusan atau decision-support system (DSS). 

Seiring berjalannya waktu, data warehouse berkembang menjadi lebih efisien. Ia berevolusi dari penyimpanan informasi pendukung platform business intelligence menjadi infrastruktur analitis luas yang mendukung berbagai macam aplikasi.

### RDBMS

Dalam model relasional, sebuah database terdiri dari banyak tabel. Sebuah tabel dibentuk dari kolom dan baris yang memuat nilai tertentu. Konsep **Relational Database Management System** (RDBMS) sendiri merupakan sistem yang mendukung adanya hubungan atau relasi antar tabel pada suatu database. Setiap tabel dihubungkan dengan tabel lainnya dengan menggunakan primary key dan foreign key. Saat ini sudah banyak jenis database yang menerapkan model RDBMS. Sebut saja MySQL, PostgreSQL, dan Microsoft SQL Server.

### NoSQL

Sesuai dengan namanya NoSQL adalah jenis basis data yang tidak menggunakan bahasa SQL dalam manipulasi datanya. Dalam penyimpanan datanya, NoSQL memiliki beberapa teknik penyimpanan yaitu: 

* Dokumen : menghubungkan setiap kunci dengan struktur data kompleks yang disebut dokumen. 
* Graph : menyimpan informasi tentang jaringan data, seperti koneksi sosial. 
* Nilai-kunci : adalah database NoSQL paling sederhana di mana setiap elemen dalam database disimpan sebagai nilai yang diasosiasikan dengan sebuah kunci. 
* Kolom : mnyimpan data yang memiliki volume besar, di mana setiap elemen data disimpan pada kolom bukan pada baris.

Beberapa database NoSQL terpopuler adalah MongoDB, CouchDB, Cassandra, Redis, Neo4J, dan Riak. Jika ingin mengetahui lebih lanjut tentang NoSQL, kunjungi [tautan](https://www.quora.com/What-are-the-main-differences-between-the-four-types-of-NoSql-databases-KeyValue-Store-Column-Oriented-Store-Document-Oriented-Graph-Database) berikut.

### Firebase Realtime Database

Sesuai namanya, “Database Realtime” adalah database yang menyimpan data yang berubah seiring waktu. Data jumlah penjualan harian, pengunjung mall setiap jam, arus lalu lintas setiap menit, atau fluktuasi saham setiap detik merupakan beberapa contoh data realtime. Data pada database realtime disimpan dalam format waktu dan nilai pada waktu yang terkait seperti gambar di bawah.

Firebase Realtime Database (FRD) adalah database berbasis cloud yang didesain khusus untuk mengelola data realtime. FRD dapat menyimpan dan melakukan sinkronisasi data secara realtime di mana setiap kali ada perubahan data terbaru, FRD langsung menyimpannya pada Cloud. FRD juga dilengkapi fitur offline di mana ketika tidak ada koneksi internet, FRD akan menyimpan data secara lokal, kemudian saat online, akan melakukan sinkronisasi ke Cloud.

### Spark

Apache Spark adalah perangkat lunak untuk pemrosesan dan analisis data berskala besar. Spark dapat digunakan dalam proses ETL (Extract, Transform, Load), data streaming, perhitungan grafik, SQL, dan machine learning. Untuk machine learning, Spark menyediakan MLlib yang berisi implementasi model machine learning seperti klasifikasi, regresi, pengklasteran, penurunan dimensi, dan pemfilteran kolaboratif.

### Big Query

BigQuery adalah data warehouse berbasis cloud untuk perusahaan yang menawarkan penyimpanan data berbasis SQL dan analisis data berukuran besar. Karena berbasis cloud dan tidak ada infrastruktur yang perlu dikelola, pengguna dapat berfokus pada pengolahan data tanpa memerlukan seorang administrator database.

## Datasets

Dataset yang telah dibersihkan dan diproses kemudian siap kita latih dengan machine learning. Satu-satunya cara untuk mengetahui apakah model machine learning kita bagus atau tidak adalah dengan mengujinya pada kasus atau data baru yang belum dikenali oleh model. Kita bisa saja membuat model dan langsung mengujinya pada tahap produksi lalu memonitor kualitasnya.

### Training Set dan Test Set

Pilihan yang lebih baik adalah dengan membagi dataset menjadi 2 bagian yaitu **data training** dan **data testing**. 

Dengan demikian, kita bisa melakukan pelatihan model pada train set, kemudian mengujinya pada test set --sekumpulan data yang belum dikenali model. Ingat bahwa membandingkan hasil prediksi dengan label sebenarnya dalam test set merupakan proses evaluasi performa model. Dengan menguji model terhadap data testing, kita dapat melihat kesalahan yang dibuat dan memperbaikinya sebelum mulai membawa model kita ke tahap produksi.

Penting untuk kita memilih rasio yang sesuai dalam pembagian dataset. Saat membagi dataset, kita perlu membuat informasi pada kedua bagian tetap berimbang. Kita tidak ingin mengalokasikan terlalu banyak informasi pada data testing agar algoritma ML dapat belajar dengan baik pada data training. Tetapi, jika alokasi data pada data testing terlalu kecil, kita tidak bisa mendapatkan estimasi performa model yang akurat.

Data testing diambil dengan proporsi tertentu. Pada praktiknya, pembagian data training dan data testing yang paling umum adalah 80:20, 70:30, atau 60:40, tergantung dari ukuran atau jumlah data. Namun, untuk dataset berukuran besar, proporsi pembagian 90:10 atau 99:1 juga umum dilakukan. Misal jika ukuran dataset sangat besar berisi lebih dari 1 juta record, maka kita dapat mengambil sekitar 10 ribu data saja untuk testing alias sebesar 1% saja.

Pada modul ini, kita akan belajar membagi dataset dengan fungsi train_test_split dari library sklearn. Perhatikan contoh kode berikut.

~~~
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1 )
~~~

Dengan fungsi **train_test_split** dari library sklearn, kita membagi array X dan y ke dalam 20% data testing (test_size=0.2 ). Misal total dataset A yang kita miliki adalah 1000 record, dengan test_size=0.2, maka data testing kita berjumlah 200 record dan jumlah data training sebesar 800 (80%).

Sebelum proses pemisahan, fungsi **train_test_split** telah mengacak dataset secara internal terlebih dahulu. Proses shuffling menjaga rasio informasi pada data training dan testing tetap berimbang.

Melalui parameter **random_state**, fungsi **train_test_split** menyediakan random seed yang tetap untuk internal pseudo-random generator yang digunakan pada proses shuffling. Umumnya, nilai yang digunakan adalah 0, atau 1, atau ada juga yang menggunakan 42. Menentukan parameter random_state bertujuan untuk dapat memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model dijalankan. Jika tidak ditentukan, maka tiap kali melakukan split, kita akan mendapatkan data train dan tes berbeda, yang juga akan membuat akurasi model ML menjadi berbeda tiap kali di-run. 

Berikut adalah contoh kode untuk memahami bagaimana penentuan random_state bekerja pada dataset:

~~~
from sklearn.model_selection import train_test_split
 
X_data = range(10)
y_data = range(10)
 
print("random_state ditentukan")
for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = 42)
    print(y_test)
 
 
print("random_state tidak ditentukan")
for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state = None)
    print(y_test)
~~~

Output:

~~~
random_state ditentukan
[3, 8, 4]
[3, 8, 4]
[3, 8, 4]
 
random_state tidak ditentukan
[9, 2, 0]
[3, 8, 5]
[1, 4, 0]
~~~

## Data Evaluation

Bayangkan ketika kita bertugas untuk mengembangkan sebuah proyek ML. Kita bimbang kala memilih model yang akan dipakai dari 10 jenis model yang tersedia. Salah satu opsinya adalah dengan melatih kedua model tersebut lalu membandingkan tingkat erornya pada test set. Setelah membandingkan kedua model, Anda mendapati model regresi linier memiliki tingkat eror yang paling kecil katakanlah sebesar 5%. Anda lalu membawa model tersebut ke tahap produksi.

Kemudian ketika model diuji pada tahap produksi, tingkat eror ternyata sebesar 15%. Kenapa ini terjadi? Masalah ini disebabkan karena kita mengukur tingkat eror berulang kali pada test set. Kita secara tidak sadar telah memilih model yang hanya bekerja dengan baik pada test set tersebut. Hal ini menyebabkan model tidak bekerja dengan baik ketika menemui data baru. Solusi paling umum dari masalah ini adalah dengan menambahkan validation set pada model machine learning.

### Train, Test, Validation Set

Validation set atau holdout validation adalah bagian dari train set yang dipakai untuk pengujian model pada tahap awal. Secara sederhana, kita menguji beberapa model dengan hyperparameter yang berbeda pada data training yang telah dikurangi data untuk validation. Lalu kita pilih model serta hyperparameter yang bekerja paling baik pada validation set. Setelah proses pengujian pada holdout validation, kita bisa melatih model menggunakan data training yang utuh (data training termasuk data validation) untuk mendapatkan model final. Terakhir kita mengevaluasi model final pada test set untuk melihat tingkat erornya.

Dalam menggunakan holdout validation, ada beberapa hal yang harus dipertimbangkan. Jika ukuran validation set-nya terlalu kecil, maka ada kemungkinan kita memilih model yang tidak optimal. Sebaliknya, ketika ukurannya terlalu besar, maka sisa data pada train set lebih kecil dari data train set utuh. Kondisi ini tentu tidak ideal untuk membandingkan model yang berbeda pada data training yang lebih kecil. Solusi untuk masalah ini adalah dengan menggunakan Cross Validation.

### Cross Validation

K-Fold Cross Validation atau lebih sering disebut cross validation adalah salah satu teknik yang populer dipakai dalam evaluasi model ML. Pada cross validation dataset dibagi sebanyak K lipatan. Pada setiap iterasi setiap lipatan akan dipakai satu kali sebagai data uji dan lipatan sisanya dipakai sebagai data latih. Dengan menggunakan cross validation kita akan memperoleh hasil evaluasi yang lebih akurat karena model dievaluasi dengan seluruh data. 

Berikut adalah ilustrasi dari K-cross validation:

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20210120135454e18a2d89e914789f3f987f3a6f2ccadd.png)