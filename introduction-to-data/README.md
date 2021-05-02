# Pengenalan Data

## Data Collecting

Tahap pertama dari proses pengerjaan proyek ML adalah data collecting, yaitu proses pengumpulan data.

>“Data is the new oil. It’s valuable, but if unrefined it cannot really be used. It has to be changed into gas, plastic, chemicals, etc to create a valuable entity that drives profitable activity; so must data be broken down, analyzed for it to have value.”
-Clive Humby, 2006-

Kutipan di atas adalah kalimat terkenal tentang data yang pertama kali disampaikan oleh Clive Humby, seorang matematikawan asal Inggris pada tahun 2006. Kutipan tersebut menjadi sangat populer setelah [The Economist](https://www.economist.com/leaders/2017/05/06/the-worlds-most-valuable-resource-is-no-longer-oil-but-data) mempublikasikan laporan tahun 2017 yang berjudul The World’s most valuable resource is no longer oil, but data.

Perangkat cerdas dan internet telah membuat data menjadi berlimpah. Banjir arus data yang terjadi di era digital mengubah sifat persaingan. Perusahaan teknologi raksasa berlomba-lomba mengumpulkan banyak data untuk meningkatkan produknya, menarik lebih banyak pengguna, menghasilkan lebih banyak data, dan seterusnya. Mereka menjangkau seluruh sektor ekonomi: Google bisa melihat apa yang ditelusuri dan dicari oleh orang-orang, Facebook bisa melihat apa yang mereka bagikan, dan Amazon mengetahui apa yang mereka beli. Mereka seolah memiliki “God’s eyes view” tentang aktivitas di pasar mereka sendiri dan sekitarnya.

Ada tiga cara yang bisa kita lakukan untuk mengumpulkan data, yaitu:

* Mengekstrasi data (misal dari internet, riset, survei, dll).
* Mengumpulkan dan membuat dataset Anda sendiri dari nol.
* Menggunakan dataset yang telah ada.

Untuk saat ini, kita akan menggunakan dataset yang sudah ada dari platform penyedia data.

Menemukan dataset yang tepat adalah salah satu langkah penting dalam proyek machine learning. Saat ini, tersedia banyak sumber data di internet yang dapat kita manfaatkan. Beberapa di antaranya yang perlu Anda ketahui adalah sebagai berikut:

**UC Irvine Machine Learning Repository**

[UCI ML Repository](https://archive.ics.uci.edu/ml/index.php) adalah kumpulan database, teori, dan generator data yang digunakan oleh komunitas ML untuk analisis algoritma machine learning. Arsip tersebut awalnya dibuat sebagai arsip ftp pada tahun 1987 oleh David Aha, seorang mahasiswa pascasarjana UC Irvine. Sejak saat itu database UCI ML Repository ini digunakan secara luas oleh mahasiswa, staf pengajar, dan peneliti di seluruh dunia sebagai salah satu sumber utama dataset machine learning.

**Kaggle Dataset**

[Kaggle](https://www.kaggle.com/datasets) adalah komunitas belajar ilmu data paling populer di dunia. Kaggle memiliki peralatan dan sumber daya yang kuat untuk membantu kita belajar data science dan machine learning. Saat ini Kaggle memiliki 50.000 lebih publik dataset, baik dataset bersifat dummy ataupun riil yang dapat Anda unduh secara bebas.

**Google Dataset Search Engine**

Pada akhir tahun 2018 Google meluncurkan [Dataset Search](https://datasetsearch.research.google.com/), sebuah mesin pencari dataset. Tools ini bertujuan untuk menyatukan ribuan repositori dataset yang berbeda agar dataset tersebut lebih mudah ditemukan oleh pengguna.

**Tensorflow Dataset**

Seperti yang telah dijelaskan pada sub-modul sebelumnya, [TensorFlow](https://www.tensorflow.org/) adalah framework open source untuk machine learning yang dikembangkan dan digunakan oleh Google. Selain menyediakan [learning resources](https://www.tensorflow.org/learn), tensorflow juga menyediakan [data resources](https://www.tensorflow.org/datasets/catalog/overview) yang cukup lengkap di library-nya mulai dari audio data, images, text, video, dan lainnya.

**US Government Data**

Bagi Anda yang tertarik untuk mempelajari fenomena yang terjadi di Amerika Serikat, pemerintah Amerika meluncurkan [data online resources](https://www.data.gov/) yang mudah diakses oleh publik. Isinya antara lain data badai, data angka kelulusan dan dropout, data hewan-hewan yang terancam punah, statistik kriminal, dan berbagai data menarik lainnya.

**Satu Data Indonesia**

Pemerintah Indonesia, melalui portal resmi [Satu Data Indonesia](https://data.go.id/) menjalankan kebijakan tata kelola data pemerintah yang bertujuan untuk menciptakan data berkualitas, mudah diakses, dapat dibagi, dan digunakan oleh Instansi Pusat serta Daerah. Data dalam portal ini dapat diakses secara terbuka dan dikategorikan sebagai data publik, sehingga tidak memuat rahasia negara, rahasia pribadi, atau hal lain sejenisnya sebagaimana diatur dalam Undang-undang nomor 14 Tahun 2008 tentang Keterbukaan Informasi Publik.

**Open Data Pemerintah Jawa Barat**

[Open data Jawa Barat](https://data.jabarprov.go.id/dataset) adalah portal resmi data terbuka milik Pemerintah Provinsi Jawa Barat yang berisikan data-data dari Perangkat Daerah di lingkungan Pemerintah Provinsi Jawa Barat. Open Data Jawa Barat ada untuk dapat memenuhi kebutuhan data publik bagi masyarakat. Data disajikan dengan akurat, akuntabel, valid, mudah diakses dan berkelanjutan. 

## Data Cleaning

Kita mungkin berpikir pekerjaan data scientist atau machine learning engineer adalah membuat algoritma, mengeksplor data, membuat analisis, dan prediksi. Padahal faktanya, seseorang yang bekerja di bidang data membutuhkan sebagian besar waktunya untuk melakukan proses data cleaning. Hasil penelitian CrowdFlower dalam 2016 Data Science Report menyatakan bahwa 3 dari 5 data scientist yang disurvei menggunakan sebagian besar waktunya untuk membersihkan dan mengatur data.

![](https://lh4.googleusercontent.com/58bq-CVvkmsUVbBK0JEov8kAuCPv9BeyqM2a6lXyQipCJAKzLec4bF2-YRW9Mfr2sWC1xOChsAGBkbnG8LluMaCoszlAPWVquFOT3CBIDpjlWaQTqByKE5GuiKFJRVBRPkMi9ZCl)

Data cleaning penting sebab proses ini meningkatkan kualitas data yang juga berpengaruh terhadap produktivitas kerja secara keseluruhan. Data yang tidak akurat bisa berpengaruh buruk terhadap akurasi dan performa model. Saat kita melakukan proses data cleaning, kita membuang data dan informasi yang tidak dibutuhkan sehingga kita akan mendapatkan data yang berkualitas. 

Data yang akurat dan berkualitas akan berpengaruh positif terhadap proses pengambilan keputusan. Pernahkah mendengar ungkapan “Garbage In - Garbage Out?” Dalam konteks machine learning,  jika input yang Anda masukkan itu buruk, sudah barang tentu hasil olah data Anda pun akan buruk. 

Data cleaning merupakan tahapan penting yang tidak boleh Anda lewatkan. Berikut adalah beberapa hal umum yang harus diperhatikan dalam proses data cleaning:

1. Konsistensi Format
Sebuah variabel mungkin tidak memiliki format yang konsisten seperti penulisan tanggal 10-Okt-2020 versus 10/10/20. Format jam yang berbeda seperti 17.10 versus 5.10 pm. Penulisan uang seperti 17000 versus Rp 17.000. Data dengan format berbeda tidak akan bisa diolah oleh model machine learning. Solusinya, format data harus disamakan dan dibuat konsisten terlebih dahulu.

2. Skala Data
Jika sebuah variabel memiliki jangka dari 1 sampai 100, pastikan tidak ada data yang lebih dari 100. Untuk data numerik, jika sebuah variabel merupakan bilangan positif, maka pastikan tidak ada bilangan negatif.

3. Duplikasi data
Data yang memiliki duplikat akan mempengaruhi model machine learning, apalagi jika data duplikat tersebut besar jumlahnya. Untuk itu kita harus memastikan tidak ada data yang terduplikasi.

4. Missing Value
Missing value terjadi ketika data dari sebuah record tidak lengkap. Missing value sangat mempengaruhi performa model machine learning. Ada 2 (dua) opsi untuk mengatasi missing value, yaitu menghilangkan data missing value atau mengganti nilai yang hilang dengan nilai lain, seperti rata-rata dari kolom tersebut (mean) atau nilai yang paling sering muncul (modus), atau nilai tengah (median).

5. Skewness Distribution
Skewness adalah kondisi di mana dataset cenderung memiliki distribusi data yang tidak seimbang. Skewness akan mempengaruhi data dengan menciptakan bias terhadap model. Apa itu bias? Sebuah model cenderung memprediksi sesuatu karena ia lebih sering mempelajari hal tersebut. Misalkan ada sebuah model untuk pengenalan buah di mana jumlah jeruk 92 buah dan apel 8 buah. Distribusi yang tidak imbang ini akan mengakibatkan model lebih cenderung memprediksi jeruk daripada apel.
Cara paling simpel untuk mengatasi skewness adalah dengan menyamakan proporsi kelas mayoritas dengan kelas minoritas. Untuk teknik lebih lanjut dalam mengatasi skewness atau imbalance data, Anda bisa membacanya di [tautan](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data) ini.

[Data Cleaning towardsdatascience](https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4)
[Data Cleaning kdnuggets](https://www.kdnuggets.com/2019/06/7-steps-mastering-data-preparation-python.html)

## Data Processing

Pada tahap ini, setelah data diambil dari sumber tertentu, ia dimasukkan pada suatu environment. Lantas data diproses agar bisa diolah oleh model machine learning. Menjalankan proses machine learning sama seperti mengajari seorang anak kecil. Ketika mengajari anak kecil membedakan antara buah apel dan buah jeruk, kita tinggal memperlihatkan buahnya dan memberi tahu mana apel dan mana jeruk. Namun demikian, komputer saat ini belum secanggih dan seintuitif itu sehingga kita perlu mempersiapkan data dengan data processing agar bisa dimengerti komputer. 

### Pandas Library

Pandas adalah sebuah library open source yang dipakai untuk menganalisis dan memanipulasi data. Pandas dibangun menggunakan bahasa pemrograman Python yang menawarkan struktur data dan operasi untuk manipulasi tabel numerik dan time series. 

Tabel numerik adalah tabel yang berisi bilangan numerik dan Tabel time series adalah tabel yang berubah seiring waktu, misalnya tabel yang memuat perubahan nilai pasar saham untuk setiap menitnya.

Berbagai jenis data yang umum dipakai dalam ML seperti CSV dan SQL dapat diubah menjadi dataframe pandas. Dataframe adalah sebuah tabel yang terdiri dari kolom dan baris dengan banyak tipe data di dalamnya. Pandas terintegrasi dengan library machine learning yang populer seperti Scikit Learn (SKLearn) dan Numpy.

Pandas mendukung banyak jenis data yang dapat dipakai dalam sebuah project machine learning. Berikut adalah beberapa contoh data yang dapat diolah dengan pandas.

* CSV
CSV adalah sebuah format data di mana elemen dari setiap baris dipisahkan dengan koma. CSV sendiri adalah singkatan dari Comma Separated Value.

* SQL
Structured Query Language adalah sebuah data yang berasal dari sebuah relational database. Format data ini berisi sebuah tabel yang memiliki format data seperti integer, string, float, dan biner.

* EXCEL
Excel adalah berkas yang didapat dari spreadsheet seperti Microsoft Excel atau Google Spreadsheet. File Excel biasanya memuat data numerik.

* SPSS
SPSS atau Statistical Package for the Social Science adalah sebuah berkas dari perangkat lunak yang biasa dipakai untuk statistik dan pengolahan data. Berkas SPSS yang disimpan memiliki ekstensi .sav.

* JSON
JSON atau Javascript Object Notation adalah salah satu format data yang menggunakan sistem Key - Value di mana sebuah nilai disimpan dengan key tertentu untuk memudahkan mencari data.

Pada kelas ini kita hanya akan menggunakan data berjenis csv.

Untuk tahapan selanjutnya Anda dapat memilih untuk menulis dan menjalankan kode pada [IBM Watson Studio](https://cloud.ibm.com/registration?cm_sp=Cloud-Home-_-LeadspaceReg-IBMCloud_CloudHome-_-LSReg) atau [Google Colaboratory](http://colab.research.google.com/). 

[Google Colaboratory](https://imam.digmi.id/post/google-colab-gratis-untuk-belajar-deep-learning/)

[Latihan dengan Watson Studio](https://jp-tok.dataplatform.cloud.ibm.com/)