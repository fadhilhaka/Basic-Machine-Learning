# Support Vector Machine

Vladimir N Vapnik, seorang Professor dari Columbia, Amerika Serikat pada tahun 1992 memperkenalkan sebuah algoritma training yang bertujuan untuk memaksimalkan margin antara pola pelatihan dan batas keputusan (decision boundary) [[10]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference). Algoritma ini kemudian dikenal luas sebagai Support Vector Machine (SVM).

Support Vector Machine adalah model ML multifungsi yang dapat digunakan untuk menyelesaikan permasalahan klasifikasi, regresi, dan pendeteksian outlier. Termasuk ke dalam kategori supervised learning, SVM adalah salah satu metode yang paling populer dalam machine learning.

Tujuan dari algoritma SVM adalah untuk menemukan hyperplane terbaik dalam ruang berdimensi-N (ruang dengan N-jumlah fitur) yang berfungsi sebagai pemisah yang jelas bagi titik-titik data input. Untuk lebih jelasnya, perhatikan gambar berikut.

![](https://lh5.googleusercontent.com/dpUhrvykVpnGCBBnT8slK6VR_Zbk_gXNDaoLWTlznzmUm5PgVQLEczplXskmIzGtYKDVYQUEK8PVvkQkdNXfbL0vhnjE4AoJGqlqa0Yh-0gnItZJET97mdSXyE4cFSFlnXTBCp3X)

Gambar di sebelah kiri menunjukkan beberapa kemungkinan bidang (hyperplane) untuk memisahkan data lingkaran dan data segitiga.

Algoritma SVM kemudian mencari hyperplane terbaik yang dapat memisahkan kedua kelas secara optimal. Seperti tampak pada gambar di sebelah kanan, sebuah hyperplane optimal berhasil dibuat dan mampu memisahkan kedua kelas sehingga memiliki margin yang maksimal.

Beberapa keunggulan Support Vector Machine antara lain:

* SVM efektif pada data berdimensi tinggi (data dengan jumlah fitur atau atribut yang sangat banyak).
* SVM efektif pada kasus di mana jumlah fitur pada data lebih besar dari jumlah sampel.
* SVM menggunakan subset poin pelatihan dalam fungsi keputusan (disebut support vector) sehingga membuat penggunaan memori menjadi lebih efisien.

## Support Vector Machine Classifier

Modul berikut akan membahas tentang SVM pada kasus klasifikasi. Untuk memahami bagaimana algoritma support vector machine atau SVM bekerja pada kasus klasifikasi, bayangkan kita memiliki sebuah kebun binatang mini. Di kebun binatang tersebut terdapat dua jenis binatang yaitu, ayam hias dan ular.

Sebagai seorang ML engineer, kita ingin mengembangkan sebuah model yang mampu membedakan antara ayam dan ular piton agar bisa menempatkan kedua hewan tersebut di kandang yang berbeda. Kita tentunya tak mau menyatukan ayam dan ular dalam satu kandang yang sama.

Kita bisa membuat sebuah model klasifikasi yang memisahkan antara kedua kelas tersebut menggunakan Support Vector Machine. Menurut Aurelien Geron dalam buku Hands on Machine Learning, SVM bekerja dengan membuat decision boundary atau sebuah bidang yang mampu memisahkan dua buah kelas. Pada masalah ini decision boundary yang mampu memisahkan kelas ayam dan kelas ular adalah sebuah garis lurus yang dapat dilihat pada gambar.

![](https://lh6.googleusercontent.com/Qt3_7GC7kNHsHD83rpELNd_-JpvzOXRS26qJVPTkEmjiYeINYhMmUJRdM_zs0WeG0jkZETTvXe9kK_LhVZ7b8BhMfF_39A2m9Wr5CzTZ8s7PCLZ3_9vjF8aFa_AJgFCuPRwqnPJo)

Pertama SVM mencari support vector pada setiap kelas. Support vector adalah sampel dari masing-masing kelas yang memiliki jarak paling dekat dengan sampel kelas lainnya. Pada contoh dataset bola basket dan bola kaki di bawah, support vector adalah bola basket dan bola kaki yang memiliki warna biru. 

![](https://lh4.googleusercontent.com/BTlqmi2wp983xOSBMJ_Atp7pLFLV20rY2g6X-gFl31d-nOUtbzlLwE2yY59_ntFp3MWv39lR7w8rHnXoBCMFygtCUxBnuHZO0DQOYK0SqNBpRTjh8CDzRTwA821HgKolFAZLwr4m)

Setelah support vector ditemukan, SVM menghitung margin. Margin bisa kita anggap sebagai jalan yang memisahkan dua kelas. Margin dibuat berdasarkan support vector di mana support vector bekerja sebagai batas tepi jalan, atau sering kita kenal sebagai bahu jalan. SVM mencari margin terbesar atau jalan terlebar yang mampu memisahkan kedua kelas.

Pada dataset bola basket dan bola kaki di atas SVM akan memilih margin di sebelah kanan karena ‘jalan’ atau margin pada gambar sebelah kanan lebih lebar dari ‘jalan’ di sebelah kiri. Oleh karena itu, gambar sebelah kanan disebut sebagai high margin classification dan gambar di sebelah kiri disebut low margin classification. 

Kembali lagi ke kasus klasifikasi ayam dan ular, sampel ayam dan ular yang berada dalam lingkaran merah adalah support vector. Kemudian kita mencari jalan terlebar dari 2 support vector. Setelah menemukan jalan terlebar, decision boundary lalu digambar berdasarkan jalan tersebut.

![](https://lh3.googleusercontent.com/TFiiUkSPWRcGsRfaolBJxB3LakpgRvgzjTWUghhDihQqIhlE2CrgZRtHl57LPonu8PdOciJmUCP-UxKfP_ZHkihsUPQWXLrwmA3xQlKkb3FZblgat8iA36Fc9cfBGsBnNE3Ftr6K)

Decision boundary adalah garis yang membagi jalan atau margin menjadi 2 bagian yang sama besar. Hyperplane adalah bidang yang memisahkan kedua kelas, sedangkan margin adalah lebar ‘jalan’ yang membagi kedua kelas.

## SVM Klasifikasi non Linier

![](https://lh3.googleusercontent.com/hIdPbEHZsfU5xJLpeRvrxIALGlGlk3h0ObkSBLf6bI_LnHaA6TTQNGY6YiAsoHje3PlpBxNc1aFAYjcxo5UP3wWsjkLLONIATP2nmRcslgO6kw3_UEKuCjSe8EB1hL8J61NmVy-R)

Data di atas merupakan data yang tidak bisa dipisahkan secara linier sehingga kita menyebutnya sebagai data non-linear. Pada data non-linear, decision boundary yang dihitung algoritma SVM bukan berbentuk garis lurus. Meski cukup rumit dalam menentukan decision boundary pada kasus ini, tapi kita juga mendapatkan keuntungan, yaitu, bisa menangkap lebih banyak relasi kompleks dari setiap data poin yang tersebar.

Untuk data seperti di atas, Support Vector Classifier menggunakan sebuah metode yaitu “kernel trick” sehingga data dapat dipisahkan secara linier. Apa itu trik kernel? Ia adalah sebuah metode untuk mengubah data pada dimensi tertentu (misal 2D) ke dalam dimensi yang lebih tinggi (3D) sehingga dapat menghasilkan hyperplane yang optimal. Perhatikan gambar berikut [[11]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference).

![](https://lh3.googleusercontent.com/Am4kJqOGDXSoPcnqKijAi3HFE3xDclJfwEhuB1QOQu3HFZF-d8QbP-QWincVH15mbVOmTx0tq3Z_LtU3Iz_JogdwbifVQ81pUyQxGt7-DW0yr5UYmr8pJJKGFR7PkpfFaWQiwVIa)

Pertama, kita perlu menghitung skor jarak dari dua titik data, misal x_i dan x_j. Skor akan bernilai lebih tinggi untuk titik data yang lebih dekat, dan sebaliknya. Lalu kita gunakan skor ini untuk memetakan data pada dimensi yang lebih tinggi (3D). Teknik ini berguna untuk mengurangi waktu dan sumber daya komputasi, terutama untuk data berjumlah besar. Hal ini juga mencegah kebutuhan akan proses transformasi yang lebih kompleks. Itulah mengapa teknik ini sering disebut sebagai trik kernel.

Seperti yang ditunjukkan gambar di atas, pemetaan titik data dari ruang 2D menjadi 3D menggunakan fungsi kernel. Titik-titik merah yang sebelumnya berada di tengah sekarang berada di dalam bidang vertikal dengan posisi lebih rendah setelah diubah menjadi ruang 3D. Titik data yang sebelumnya sulit dipisahkan, sekarang dapat dengan mudah dipisahkan dengan teknik kernel.

Animasi yang dibuat oleh Udiprod [[12]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) berikut dapat membantu kita untuk melihat bagaimana data 2 dimensi -seperti pada gambar sebelumnya- diubah ke dalam ruang 3 dimensi, sehingga data dapat dipisahkan secara linier.

![](https://lh6.googleusercontent.com/ZU_JlI7LVOxLWiVv1RRiRWIdWCBvjBgbQv7T6az0s-rA7APYFsljxKDiE9AJD-GXEAX2lGVMa9T40m-0AyhAFmTLR3fMciC6NfWJuh6aX3wxrFmCKIhKPcWTxdbJN5i--UH1jmWE)

![](https://lh6.googleusercontent.com/DmrmlulXMweCOyfl9BLK97j5s6FT2M1jG4rC0OO0Qpom7zGPa0wavObxt7HQtVYhstuSgeZI--NhkeyXMMw_dEsiSi32x-KlqqobYa6EhHP0jQEDvYmo-fFM6uFhaDHAp1JFoho_)

Data di atas adalah data 1 dimensi dengan 2 buah kelas yaitu dokter dan polisi. Data di atas bukan data linier karena kita tak dapat menggambar satu garis lurus untuk memisahkan 2 kelas yang ada. Bagaimana cara kita bisa menggambar garis lurus yang bisa memisahkan 2 kelas tersebut? Betul, kita akan menggunakan trik kernel untuk mengubah data tersebut ke dalam dimensi yang lebih tinggi seperti ke dalam bidang 2 dimensi.

![](https://lh4.googleusercontent.com/ZEJTdIKyLiq7oPkCR1O8J3RljOP7G2XSVaDJpBR8EzdiRZT6T_XIFVvKcslNLMuedyCPP-F6sx7j0zOHOxC7_gZdUTjq81nulyI7cF8tSIKAZFWvTtERYReUHO7C1ntyhjbi-Rms)

Ketika data sudah diubah ke dalam bidang 2 dimensi, sebuah garis lurus bisa digambar untuk memisahkan 2 kelas. Trik kernel menggunakan fungsi matematis yang bisa mengubah data dari dimensi tertentu ke dimensi yang lebih tinggi sehingga kelas-kelas pada data dapat dipisah secara linier.

Berikut adalah beberapa fungsi kernel:

* [Linear](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
* [RBF (Radial Basis Function) atau Gaussian kernel](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py)
* Polinomial
* Sigmoid

Untuk melihat bagaimana pemisahan data dan seperti apa bentuk hyperplane pada masing-masing fungsi kernel, perhatikan gambar yang dibuat oleh H. Marius [[13]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) berikut. 

![](https://lh3.googleusercontent.com/Nxovq_I6OMb-EuHUHcxayhmuq4ZciqPNdGBMxiJ32ZSoxTY39_wb7O7vkU85dZExlXLQsFEDwXS7q-FspQ57led8B6vNK0zWdaWs_D81Usgalh1ymyKYb-HvNXiKxao_J30vnaaR)

Salah satu contoh aplikasi SVM dalam kehidupan sehari-hari adalah fitur deteksi wajah. Pengenalan wajah telah berkembang menjadi topik penelitian utama dalam bidang computer vision. Sistem pengenalan wajah memiliki banyak fitur/kelas (individu) dan hanya memiliki beberapa data gambar (sampel) per orang, bahkan kadang hanya ada satu sampel pelatihan untuk setiap orang. Permasalahan dengan jumlah fitur lebih banyak dari jumlah sampel ini efektif jika dipecahkan dengan algoritma SVM.

Aplikasi lain SVM yang juga menarik adalah di bidang bioinformatika. Dalam bidang komputasi biologi, deteksi homologi jarak jauh pada protein (protein remote homology detection) adalah permasalahan yang umum. Metode paling efektif untuk menyelesaikan permasalahan ini adalah dengan SVM. Dalam beberapa tahun terakhir, algoritma SVM telah diterapkan secara luas untuk proses deteksi ini. Fungsi kernel pada SVM digunakan untuk mengidentifikasi sekuens biologis dan membantu menemukan kesamaan antara urutan protein yang berbeda. 

[Fungsi matematik di balik kernel](https://data-flair.training/blogs/svm-kernel-functions/)

## SVM untuk Klasifikasi Multi-kelas

SVM sejatinya merupakan binary classifier atau model untuk klasifikasi 2 kelas. Namun, SVM juga dapat dipakai untuk klasifikasi multi-kelas menggunakan suatu teknik yaitu “one-vs-rest”. 

Pada masalah klasifikasi multi-kelas, SVM melakukan klasifikasi biner untuk masing-masing kelas. Model kemudian memisahkan kelas tersebut dari semua kelas lainnya, menghasilkan model biner sebanyak jumlah kelasnya. Untuk membuat prediksi, semua proses klasifikasi biner dijalankan pada tahap uji. 

Sebagai contoh, jika ada 3 buah kelas: donat, ayam, dan burger, SVM akan melakukan 3 kali klasifikasi. Pertama, membangun pemisah antara kelas donat dan kelas bukan donat. 

Kemudian membangun pemisah antara kelas ayam dan kelas bukan ayam, lalu pemisah antara kelas burger dan bukan kelas burger. Teknik inilah yang disebut dengan “One-vs-Rest”.

![](https://lh4.googleusercontent.com/-d0sDyan39BiofAC7ojoa3B-4K8Uxr11vAvIrBWskj_g27twVuGixg66Idwnzs8bGFSEHE4mnhOCIFz3zNIuPUlsYGbOVnbMnoAxQWO_IxO_93Kj-IndKOp1_VHrk1PCuWE9c5Ak)