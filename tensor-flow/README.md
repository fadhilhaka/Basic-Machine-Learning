# Tensor Flow

TensorFlow(TF) adalah end-to-end open source platform yang dikembangkan oleh Google Brain dan sangat populer untuk pengembangan machine learning berskala besar. Ia memiliki ekosistem tools, library, dan sumber daya komunitas yang komprehensif dan fleksibel, yang memungkinkan para peneliti dan pengembang membangun dan menerapkan (deploy) aplikasi machine learning dengan mudah.

Pada awalnya TensorFlow digunakan untuk menjalankan komputasi numerik kompleks pada riset AI dan machine learning di internal Google. Dalam perkembangannya kemudian, TensorFlow menjadi tools efektif dan powerful untuk menyelesaikan permasalahan deep learning di kalangan  masyarakat luas. 

Jeff Hale, seorang pegiat data science melakukan riset pada akhir tahun 2018 tentang Deep Learning Framework Power Scores [[1]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference). Riset tersebut menggunakan 11 sumber data pada 7 kategori yang berbeda untuk mengukur penggunaan dan popularitas framework dan ketertarikan pengguna. Hale menghitung dan melakukan visualisasi data risetnya pada platform [kaggle](https://www.kaggle.com/discdiver/deep-learning-framework-power-scores-2018), kemudian mempublikasikan hasil risetnya di [medium](https://towardsdatascience.com/deep-learning-framework-power-scores-2018-23607ddf297a). Dari riset tersebut, TensorFlow menempati urutan pertama seperti ditunjukkan pada gambar berikut.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/2021041609383927e174a211c642bf437b1405ac70efe4.jpeg)

Tensorflow dirilis pertama kali pada akhir tahun 2015 [[2]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference). Versi stabil TensorFlow mulai dapat digunakan pada tahun 2017, sementara versi 2.x dirilis untuk publik pada bulan September 2019 lalu. Saat ini, TensorFlow digunakan secara luas pada berbagai produk keluaran Google yang kita gunakan sehari-hari, antara lain, Google Cloud Speech, Google Photos, Gmail, dan Google Search. Untuk melihat video singkat dari tim Google Brain tentang TensorFlow saat pertama kali diluncurkan, silakan buka [tautan berikut](https://www.youtube.com/watch?v=oZikw5k_2FM).

## Bagaimana TensorFlow Bekerja?

Tim Google Brain dalam makalahnya yang berjudul “Tensorflow: A System for Large Scale Machine Learning”  menjelaskan bahwa pada TensorFlow, data dimodelkan sebagai tensor (array berdimensi-n) dengan elemen yang memiliki salah satu dari tipe data int32, float32, atau string. Secara alami, tensor mewakili masukan untuk operasi matematika dalam berbagai algoritma machine learning. Sebagai contoh, perkalian matriks membutuhkan dua buah tensor 2-D dan akan menghasilkan tensor 2-D juga. 

TensorFlow menggunakan grafik aliran data untuk mewakili seluruh proses komputasi dan state dalam algoritma machine learning, termasuk operasi matematika. Parameter, pembaruan aturan (update rules), dan masukan preprocessing. Grafik aliran data merepresentasikan komunikasi antara sub komputasi secara eksplisit sehingga mudah untuk menjalankan penghitungan independen secara paralel. Berikut adalah skema aliran data TensorFlow untuk pipeline pelatihan berisi subgraf untuk membaca data masukan, preprocessing, training, dan checkpointing [[2]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference).

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/2021041609383823c2d6b3f6f553e39fffcff74726b92a.jpeg)

Secara sederhana, cara kerja TensorFlow dapat kita gambarkan sebagai berikut. Pertama kita tentukan sebuah komputasi graf untuk diselesaikan seperti gambar di bawah. Variabel x, y, dan angka 2 pada kotak berwarna jingga adalah input atau masukan, sedangkan operasi dilambangkan dalam lingkaran berwarna biru. Selanjutnya TensorFlow akan mengeksekusinya secara efisien dengan kode C++ yang teroptimasi. Hasilnya, operasi matematis gambar di bawah memberikan output berupa sebuah persamaan kuadrat.

Secara sederhana, cara kerja TensorFlow dapat kita gambarkan sebagai berikut. Pertama kita tentukan sebuah komputasi graf untuk diselesaikan seperti gambar di bawah. Variabel x, y, dan angka 2 pada kotak berwarna jingga adalah input atau masukan, sedangkan operasi dilambangkan dalam lingkaran berwarna biru. Selanjutnya TensorFlow akan mengeksekusinya secara efisien dengan kode C++ yang teroptimasi. Hasilnya, operasi matematis gambar di bawah memberikan output berupa sebuah persamaan kuadrat.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/202104160938380ab8aeb723350605cd6afc0381983c8f.jpeg)

TensorFlow mampu membagi grafik di atas menjadi beberapa bagian dan menjalankannya secara paralel menggunakan CPU dan GPU. Dengan kemampuan tersebut, kita dapat melakukan pelatihan sebuah model jaringan saraf yang sangat besar hingga mencakup ratusan server, secara paralel. 

## Keunggulan TensorFlow

TensorFlow menggunakan Python sebagai front-end API-nya sehingga mudah dan nyaman digunakan, bahkan oleh pemula sekalipun. Perlu dicatat bahwa TensorFlow ditulis dan dieksekusi dengan bahasa C++ yang berkinerja tinggi. Beberapa keunggulan TensorFlow antara lain:

* Bisa dijalankan di hampir semua platform: GPU, CPU, dan TPU (TensorFlow Processing Units) yang secara khusus dimanfaatkan untuk mengerjakan matematika tensor. 
* Memberikan performa terbaik dengan kemampuan melakukan iterasi dan melatih model secara cepat sehingga mampu menjalankan lebih banyak eksperimen.
* Skalabilitas komputasi yang tinggi pada kumpulan data yang sangat besar.
* Pembuatan model yang mudah dengan beberapa level abstraksi sesuai kebutuhan.
* Menyediakan jalur langsung ke produksi, baik itu pada server, perangkat mobile atau web sehingga memudahkan kita melakukan pipeline machine learning hingga ke level produksi.

Ekosistem TensorFlow 2.x dari training hingga produksi beserta framework yang digunakan untuk setiap prosesnya dapat kita pelajari pada gambar berikut [[3]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference).

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20210416093839b2f5741e10429c741a378562c4001728.jpeg)

TensorFlow menyediakan semua tools dan library yang kita butuhkan untuk proyek machine learning dari tahap training model hingga tahap produksi. Itulah alasan mengapa TensorFlow disebut sebagai end-to-end platform untuk machine learning. Untuk mempelajari lebih lanjut tentang TensorFlow Anda dapat membuka [tautan berikut](https://www.tensorflow.org/). Pelajari juga tutorial dan panduan menggunakan TensorFlow pada [tautan ini](https://www.tensorflow.org/tutorials) dan [tautan berikut](https://www.tensorflow.org/guide).