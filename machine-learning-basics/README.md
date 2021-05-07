# Dasar-Dasar Machine Learning

## Machine Learning Workflow

Dalam sebuah project machine learning ada tahapan-tahapan yang perlu dilalui sebelum project tersebut bisa diimplementasi di tahap produksi. Tahapan-tahapan yang dimaksud menurut buku Hands on Machine Learning karya Geron [[4]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) dapat dirangkum dalam diagram berikut.

![](https://lh5.googleusercontent.com/AQUB5ZURSkkYQqpDa8nssTekhfLtnzzIpK2QSb0dpZn3C3joj-Jz0YOTOb4T2P6BHWTei1vKvsMBosk6ASQARkv6ol_qadWl8qzdq8RDolR-ZV7uVWONUBXc7rL0mU7paKSimBq1)

Tahapan dalam diagram ini bersifat iteratif yang berarti prosesnya bisa berulang sesuai kebutuhan. Anda mungkin perlu untuk mengevaluasi ulang proses yang Anda jalankan dan kembali ke langkah sebelumnya, kapan saja dibutuhkan selama prosesnya.

### Proses pengumpulan data

Pada modul [Intro to Data](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/introduction-to-data) Anda telah mengetahui beberapa sumber dataset untuk model machine learning. Dari sumber-sumber tersebut, Anda dapat memilih, mengunduh dan menggunakan dataset yang sesuai dengan kebutuhan Anda kapan saja. Proses ini relatif mudah. Tantangannya adalah memilih dataset yang tepat untuk model Anda. 

Tetapi jika Anda adalah seorang Machine Learning Engineer pada sebuah perusahaan yang bertugas untuk membangun model ML untuk keperluan perusahaan, tentu proses pengumpulan datanya tidak semudah Anda mengunduh dataset yang sudah jadi. Anda perlu mengumpulkan dan mengekstrak sendiri data dari berbagai sumber misalnya database, file, data sensor, dan sumber lainnya. 

Pada tahap ini Anda juga perlu berurusan dengan berbagai jenis tipe data dari mulai structured data (seperti excel file atau database SQL), sampai unstructured data (seperti text file, email, video, audio, gambar, data sensor, dan lainnya). 

Menurut [Gartner](https://www.gartner.com/imagesrv/media-products/pdf/quadient/Quadient-1-69GN2HQ.pdf) lebih dari 80% data perusahaan adalah unstructured data.

### Exploratory Data Analysis

Exploratory data analysis atau EDA bertujuan sebagai analisa awal terhadap data dan melihat bagaimana kualitas data untuk meminimalkan potensi kesalahan di kemudian hari. 

Pada proses ini dilakukan investigasi awal pada data untuk menemukan pola, anomali, menguji hipotesis , memahami distribusi, frekuensi, hubungan antar variabel, dan memeriksa asumsi dengan teknik statistik dan representasi grafik. 

Pada umumnya EDA dilakukan dengan tiga cara, yaitu univariate analysis, bivariate analysis, dan multivariate analysis.

Univariate analysis adalah analisis deskriptif yang memeriksa pola dengan satu variabel pada modelnya. Sedangkan bivariate analysis memiliki dua variabel pada modelnya. Multivariate analysis merupakan analisis deskriptif yang memeriksa pola dalam data multidimensi dengan membertimbangkan dua atau lebih variabel. Karena multivariate analysis mempertimbangkan lebih banyak variabel, ia dapat memeriksa fenomena yang lebih kompleks dan menemukan pola data yang mewakili dunia nyata dengan lebih akurat. 

Exploratory data analysis: [tautan 1](https://www.stat.cmu.edu/~hseltman/309/Book/chapter4.pdf), [tautan 2](https://www.kite.com/blog/python/data-analysis-visualization-python/), [tautan 3](https://datascienceguide.github.io/exploratory-data-analysis), [tautan 4](https://www.youtube.com/watch?v=zHcQPKP6NpM)

### Data preprocessing

Data preprocessing adalah tahap di mana data diolah lebih lanjut sehingga menjadi siap dipakai dalam pengembangan model ML. Dengan kata lain, proses ini mengubah dan mentransformasi fitur-fitur data ke dalam bentuk yang mudah diinterpretasikan dan diproses oleh algoritma machine learning. Termasuk di dalam data preprocessing adalah proses data cleaning dan data transformation.

Beberapa hal yang bisa dilakukan dalam proses data cleaning adalah: penanganan missing value, data yang tidak konsisten, duplikasi data, ketidakseimbangan data, dll. 

Sementara beberapa hal yang bisa dilakukan untuk proses transformasi data adalah: scaling atau merubah skala data agar sesuai dengan skala tertentu, standarisasi, normalisasi, [mengonversi data menjadi variabel kategori](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html), dan sebagainya. 

Tidak lupa pula, proses train-test split yang pernah kita pelajari pada modul sebelumnya juga merupakan bagian dari data preprocessing.

### Model selection

K. P. Murphy dalam bukunya yang berjudul Machine Learning: a Probabilistic Perspective [[16]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) menuliskan kalimat berikut: 

>“When we have a variety of models of different complexity (e.g., linear or logistic regression models with different degree polynomials, or KNN classifiers with different values of K), how should we pick the right one?”

Berangkat dari pertanyaan ini, menentukan model yang sesuai dengan data merupakan tahapan yang penting dalam machine learning workflow.

Jie Ding, et al dalam tulisannya “Model Selection Techniques -An Overview” [[17]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) menyatakan bahwa tidak ada model yang cocok secara universal untuk data dan tujuan apa pun. 

Pilihan model atau metode yang tidak tepat dapat menyebabkan kesimpulan yang menyesatkan atau performa prediksi yang mengecewakan. Sebagai contoh, saat kita memiliki kasus klasifikasi biner kita perlu mempertimbangkan apa model terbaik untuk data kita, apakah logistic regression atau SVM classifier. 

Setelah kita menentukan metode yang cocok untuk data yang ada, kita perlu mengubah hyperparameter untuk mendapatkan performa terbaik dari model. Mengubah nilai hyperparameter saat kita menjalankan algoritma ML akan memberikan hasil atau performa model yang berbeda. 

Proses menemukan performa terbaik model dengan pengaturan hyperparameter yang berbeda ini juga disebut model selection.

Dengan demikian, dalam konteks machine learning, model selection bisa berarti dua hal: pemilihan learning method atau algoritma ML dan pemilihan hyperparameter terbaik untuk metode machine learning yang dipilih. 

### Model Evaluation

Setelah mengutak-atik model Anda dengan hyperparameter yang berbeda, akhirnya Anda mendapatkan model yang kinerjanya cukup baik. Langkah selanjutnya adalah mengevaluasi model akhir pada data uji. 

Sederhananya, langkah evaluasi model dapat dijabarkan sebagai berikut: memprediksi label pada data uji, menghitung jumlah prediksi yang salah (eror) kemudian membandingkannya dengan data label yang kita miliki. Dari data perbandingan ini kita dapat menghitung akurasi atau performa model.

Pada prinsipnya proses model evaluation adalah menilai kinerja model ML pada data baru, yaitu data yang belum pernah “dilihat” oleh model sebelumnya. Evaluasi model bertujuan untuk membuat estimasi generalisasi eror dari model yang dipilih, yaitu, seberapa baik kinerja model tersebut pada data baru. Idealnya, model machine learning yang baik adalah model yang tidak hanya bekerja dengan baik pada data training, tapi juga pada data baru. Oleh karena itu, sebelum mengirimkan model ke tahap produksi, Anda harus cukup yakin bahwa performa model akan tetap baik dan tidak menurun saat dihadapkan dengan data baru.

### Deployment

Ketika model dievaluasi, model siap untuk dipakai pada tahap produksi. Caranya adalah dengan menyimpan model yang telah dilatih dari tahap preprocessing hingga pipeline prediksi. Kemudian deploy model tersebut ke tahap produksi untuk membuat prediksi dengan memanggil kode predict()-nya.

Geron [4] memberikan contoh ilustrasi model deployment seperti tampak dalam gambar berikut:

![](https://lh6.googleusercontent.com/dMFsxYnLOuUSNUXWWud42Xqx9MnU02lJ_rBCzruOjwTE9cj-QGWLzPQOVu-QMunLeuR1RFmF21TgGXOAi0oBlgRmRZagRNbBvBamnN0Vx0t0OoUKnT2aUWxwZBCaF1EGDjrXN95e)

Misalnya sebuah model regresi untuk menentukan harga rumah akan digunakan dalam situs web. Pengguna akan mengetik beberapa data tentang lokasi yang diinginkan dan mengeklik tombol “Prediksi Harga”. Proses ini akan mengirimkan query yang berisi data ke server, kemudian meneruskannya ke aplikasi web Anda. Terakhir, kode akan memanggil fungsi predict() untuk memberikan hasil prediksi pada Anda. 

### Monitoring

Model yang telah dipakai dalam tahap produksi masih harus tetap dimonitor untuk menjaga kualitasnya. Pada tahap produksi model bisa saja menemukan data yang tidak dikenali sehingga performa model dapat menurun. 

Contoh kasus misalnya, jika model Anda merupakan sistem rekomendasi yang menyarankan produk untuk pengguna, maka untuk memantau performa model bisa dengan cara menghitung jumlah produk rekomendasi yang terjual tiap hari. Jika angka ini turun (dibandingkan dengan produk yang tidak direkomendasikan), maka kemungkinan model kita perlu dilatih ulang menggunakan data-data baru. 

Jika Anda bekerja dengan model machine learning yang datanya terus berubah, Anda perlu melakukan update pada dataset dan melatih ulang model Anda secara reguler. Atau, Anda perlu membuat sistem yang dapat membuat proses update ini berjalan secara otomatis. 

## Machine Learning Use Case

### ML dan Business Intelligence

Business Intelligence atau sering disebut BI telah menjadi bidang keahlian yang penting dalam Data Analytics. BI mengacu pada penggunaan berbagai alat dan teknologi dalam proses pengumpulan, analisis, dan interpretasi data bisnis. Tujuan utama business intelligence adalah untuk memberikan informasi dan analisis yang berguna untuk membantu proses pengambilan keputusan dan strategi perusahaan. 

Sebagai contoh, mari kita perhatikan studi kasus pada sebuah perusahaan produsen coklat. Dari data tahunan yang dikumpulkan, perusahaan tersebut melihat persediaan coklat selalu habis di bulan Februari. Dari hasil analisis data lampau tersebut diketahui bahwa penjualan coklat meningkat di bulan tersebut karena adanya Hari Valentine. Maka melihat pola di masa lampau tersebut, perusahaan dapat mengambil keputusan di tahun berikutnya untuk meningkatkan produksi coklat agar meraih lebih banyak profit.

Sedikit berbeda dengan machine learning. Machine learning adalah teknik yang bisa mendeteksi pola dari sekumpulan data. Pada ML jenis regresi kita membuat model dengan data di masa lampau dan kita menggunakan data tersebut untuk memprediksi apa yang akan terjadi di masa mendatang. 

Jadi, kita bisa menyimpulkan bahwa BI adalah bidang yang menjelaskan apa yang terjadi di masa lampau dan analisis yang menyebabkan hal tersebut terjadi. BI memungkinkan data untuk dipahami dalam perannya untuk kepentingan bisnis dan dengan teknik visualisasi, kita menggunakan BI untuk membuat keputusan-keputusan bisnis. Sedangkan ML adalah bidang yang mencoba memprediksi apa yang akan terjadi di masa mendatang dari pola data di masa lalu.

Berikut adalah perbedaan antara BI dan ML yang dirangkum oleh Booz Allen Hamilton:

![](https://lh6.googleusercontent.com/PNivdR7Svm9gosY6hecaawIEuJAVoLBcgUES3XheK6kBk2k8Rcl8LL__6YIBh1_5n_-__JiwCarzDoA7cWFo8eyqvke-ULMIWMKz1L9WiNSn0Blg4PbVVxV71cySyHyrY1ubQ-i2)

### ML di Bidang Data Analytic

Di era digital saat ini perusahaan menghasilkan sejumlah besar data dari berbagai sumber. Apakah itu dari sistem perusahaan, sosial media, smartphone atau komputer klien, sensor dan instrumen iOT, dan sumber lainnya. Data analytics adalah istilah luas yang mengacu pada penggunaan berbagai teknik untuk menemukan pola dalam data. Ia adalah proses di mana data diubah menjadi informasi berharga dan prediksi untuk masa mendatang. Data analytics memungkinkan kita untuk menjelaskan apa yang terjadi di masa lalu, mendapatkan informasi penting tentang masa kini, dan membuat prediksi tentang masa depan.

Data analytics bukanlah bidang yang baru, ia telah lama digunakan dalam dunia bisnis. Data analytics di masa lalu contohnya adalah penggunaan statistik untuk memperoleh rata-rata penjualan dalam jangka waktu tertentu atau menentukan tren penjualan di daerah-daerah tertentu. Seiring berjalannya waktu, bidang data analytics terus dikembangkan untuk memenuhi kebutuhan analisis perusahaan yang juga terus berkembang.

Hingga saat ini, data analytics dan machine learning hampir selalu bekerja berdampingan untuk menyelesaikan permasalahan industri. Analis jadi bisa melihat lebih dalam suatu masalah berdasarkan data yang ada, lalu menentukan apakah masalah tersebut bisa diselesaikan dengan machine learning atau tidak. Setelah masalah diidentifikasi, maka peran seorang machine learning developer adalah mengimplementasi, mulai dari mengumpulkan data, memilih model yang sesuai, melakukan deployment, dan memonitor model tersebut. Dengan cara inilah machine learning membantu proses analisis data dan pengambilan keputusan di industri.

## Overfitting dan Underfitting

Salah satu cara untuk mengetahui apakah sebuah model underfit atau overfit adalah dengan membagi dataset menjadi train set dan test set. Setelah data tersebut dibagi, kita akan melakukan pengembangan model hanya dengan data training. Hasil pengujian model terhadap data testing dapat memberitahu kita apakah model kita underfit atau overfit.

### Overfitting

Bayangkan suatu saat Anda berkunjung ke suatu kota, kemudian Anda mengalami kejadian tidak menyenangkan seperti dicopet di dalam angkutan umum. Anda kemudian mungkin akan berpikir bahwa semua angkutan umum di kota tersebut tidak aman. Kadang-kadang, kita melakukan generalisasi yang berlebihan terhadap sesuatu.

Demikian halnya pada mesin yang juga bisa terjebak pada persepsi generalisasi yang sama. Dalam machine learning, kondisi ini disebut sebagai overfitting.

Overfitting terjadi ketika model memiliki prediksi yang terlalu baik pada data training, namun prediksinya buruk pada data testing. Ketika sebuah model overfit, model tidak dapat melakukan generalisasi dengan baik sehingga akan membuat banyak kesalahan dalam memprediksi data-data baru yang ditemui pada tahap produksi.

Contoh kasus adalah sebuah model machine learning untuk mengenali gambar anjing. Sebuah model yang overfit akan sangat menyesuaikan dengan dataset. Nah, di dataset mayoritas dari gambar anjing adalah anjing berwarna hitam. Maka model akan berpikir bahwa setiap hewan yang berwarna hitam adalah anjing. Ketika model tersebut dipakai untuk memprediksi sebuah gambar kucing dan kuda berwarna hitam, maka prediksinya adalah anjing.

Sebelum men-deploy model ML ke tahap produksi, ada teknik sederhana untuk mengecek apakah model overfit atau tidak. Pada model klasifikasi jika akurasi model pada data training tinggi dan data testing rendah, maka model yang Anda kembangkan overfitting. Pada model jenis regresi, jika model membuat kesalahan yang tinggi pada data testing maka model tersebut overfitting.

Beberapa cara untuk menghindari overfitting yaitu:

* Memilih model yang lebih sederhana, contohnya pada data yang memiliki pola linier menggunakan model regresi linear daripada model decision tree.
* Mengurangi dimensi data contohnya dengan metode PCA.
* Menambahkan data untuk pelatihan model jika memungkinkan.

### Underfitting

Underfit terjadi ketika model terlalu sederhana dan tidak mampu untuk menyesuaikan pola yang terdapat pada data latih.

Sebuah model dapat dikatakan underfit jika memiliki eror yang tinggi pada data training. Underfitting menandakan bahwa model tersebut belum cukup baik dalam mengenali pola yang terdapat pada data latih. Misalnya ketika sebuah model dilatih pada data latih yang memiliki 50 sampel coklat dan 50 sampel kacang. Setelah pembelajaran dengan data latih, model malah mengenali pada data latih terdapat 90 sampel coklat dan 10 sampel kacang.

Pada kasus klasifikasi, underfitting ditandai ketika model memiliki akurasi yang rendah pada data training. Pada kasus regresi, underfitting terjadi ketika model memiliki tingkat eror yang tinggi.

Pada ilustrasi regresi di bawah, model di sebelah kiri belum menyesuaikan dengan baik terhadap pola yang terdapat pada data. Bandingkan dengan model di sebelah kanan.

![](https://lh6.googleusercontent.com/ItYEO3wMgY4c9_apMBfl90F4HhnlG8o3VvTge1LhRrDOYzFoYqQ4wab4meWDpIUvltz1iO3pv6fG__LXdVE0csurZ6disoONd3nV_cLulhLdTZsr7RSGG-CRjrDzAEzXe-yGczrm)

Cara menghindari underfitting adalah dengan menyeleksi model atau meningkatkan performa dengan tuning hyperparameter. 

Kualitas data juga sangat mempengaruhi dataset. Model machine learning yang sangat kompleks sekalipun tidak akan memiliki performa yang baik jika data yang digunakan memiliki kualitas yang buruk. Ingat prinsip: “Garbage in, garbage out”.

### Good Fit

Sebuah model good fit akan memprediksi lebih baik dan membuat lebih sedikit kesalahan di tahap produksi. Contoh dari model yang tidak good fit seperti di bawah.

![](https://lh3.googleusercontent.com/TwY6U9_TdQvQyQTxrSav1LfjUaWqRMNVmh-WFY95Ml80py1WjxJ2TivZOdpPsIONNg6qTuJDglz5yelppuzZJgjSlNZzpy0HnLBkGDk2e1fJic5h0mvDbBXRa6FzLqZ0fkz7WJiv)

Sebuah model membuat banyak kesalahan dalam memprediksi gambar di atas. Seperti sebuah kaos kaki dan anjing yang diprediksi sebagai gajah india, dan seekor paus yang diprediksi sebagai gajah afrika.

Berikut sebuah tabel yang membandingkan model yang underfitting, good fit, dan overfitting pada masalah regresi dan klasifikasi.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/202101201406554a7694d254a6365a2629fa5b2ae46ebe.png)

## Model Selection

Sebuah model machine learning memiliki parameter yang dapat di tuning. Contohnya ketika kamu memasukkan parameter “n_cluster” pada model K-Means.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/202004301732466d6c95f7b7df543d9b49b6605663c731.png)

Ketika mengembangkan model K-Means seperti di atas, mengubah parameter -- dalam hal ini memilih jumlah n_cluster -- merupakan bentuk dari tuning parameter.

Tuning Parameter adalah istilah yang digunakan untuk meningkatkan performa model machine learning. Pada model K-means di atas, jumlah cluster yang kurang atau terlalu banyak akan menyebabkan hasil pengklasteran kurang optimal. Tuning parameter dalam hal ini adalah bereksperimen mencari parameter terbaik untuk model K-Means tersebut.

Berbagai model machine learning dari library SKLearn memiliki parameter-parameter yang bisa kita ubah untuk meningkatkan performa dari sebuah model tersebut. Contohnya pada decision tree terdapat beberapa parameter seperti di bawah.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/2020043017315919187a186828fe71570a1bcc95e1e5ac.png)

[Jenis model machine learning dan parameter setiap model](https://scikit-learn.org/stable/index.html)

**Grid Search**

Grid search memungkinkan kita menguji beberapa parameter sekaligus pada sebuah model. Contohnya kita bisa menguji beberapa jumlah cluster untuk sebuah model K-Means dan melihat bagaimana performa model K-Means terhadap nilai K yang berbeda.

[Simple Linear Regression for Salary Data](https://www.kaggle.com/vivinbarath/simple-linear-regression-for-salary-data)

## Menambahkan/Mengurangi Fitur

Andrew Ng, seorang profesor kecerdasan buatan dari Stanford dan pencetus Google Brain mengatakan:

>“Menciptakan fitur-fitur yang baik adalah pekerjaan yang sulit, memakan waktu, dan membutuhkan pengetahuan seorang pakar di bidang terkait. Machine learning terapan pada dasarnya adalah rekayasa fitur.”

### Binning

Binning adalah pengelompokan nilai sesuai dengan batas besaran yang ditentukan. Pada binning, data dikelompokkan dalam tiap ‘bin’ sesuai dengan nilai yang cocok dengan bin tersebut. Bin sederhananya adalah sebuah kategori yang menampung nilai-nilai tertentu.

Ada beberapa jenis binning di mana salah satu contohnya adalah binning jarak. Pada binning jarak, nilai-nilai dari sebuah atribut akan dikategorikan ke dalam jumlah bin tertentu yang memiliki interval sama besar. Pada gambar dibawah dapat dilihat contoh kumpulan nilai yang dibagi menjadi 4 bin, 8 bin, dan 16 bin.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20200430175945f4cc1fc6bf54a840308a61a945d1ffd6.png)

Binning bisa membantu untuk menghindari overfitting. Namun binning juga mengorbankan informasi yang terkandung dari sebuah atribut sehingga, penggunaanya perlu dilakukan dengan teliti. Di bawah adalah contoh untuk melakukan binning pada dataframe Pandas.

~~~
data[‘bin’] = pd.cut(data[‘value’], bins=[0, 30,  70, 100], labels=[“Low”, “Mid”, “High”])
 
    value    bin
0      13    Low
1      25    Low
2      32    Mid
3      94    High
4      49    Mid
~~~

Mari kita ulas aplikasi binning pada model linear seperti yang dituliskan oleh S. Guido dan A. C. Muller dalam bukunya [[1]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference). Seperti yang kita tahu, model linear hanya bisa menghasilkan relasi linear, yang mana hasil regresinya merupakan garis pada fitur tunggal. Di sisi lain, model decision tree dapat menghasilkan model yang lebih kompleks dari data, meskipun model ini sangat bergantung pada representasi data. Salah satu cara untuk membuat model linear lebih efektif dalam data kontinyu adalah dengan melakukan proses binning pada fitur.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/202104161403419af1d9ab5257be4008c76c468c0a1f28.jpeg)

Gambar di atas menunjukkan perbandingan regresi linear dan decision tree pada dataset gelombang. Dengan teknik binning, kita membagi jangkauan input untuk fitur (dalam hal ini dari -3 ke 3) ke dalam angka/bin yang tetap, misal 10. Titik-titik data kemudian direpresentasikan ke dalam bin yang telah ditentukan tadi. Ada beberapa cara untuk menentukan batas bin, misalnya dengan membuat tepi bin berjarak sama, atau menggunakan kuartil data. Pada library sklearn, kedua teknik ini diimplementasikan dengan KBinsDiscretizer.

Jika kita melakukan proses binning pada data gambar di atas, maka hasilnya akan seperti berikut:

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20210416140340ba962d5c4a5064232aa48359baf4f6ea.jpeg)

Binning: [tautan 1](https://www.coursera.org/lecture/data-analysis-with-python/binning-in-python-T8z3M), [tautan 2](https://developers.google.com/machine-learning/data-prep/transform/bucketing), [tautan 3](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/group-data-into-bins)

### Splitting

Memisahkan sebuah atribut menjadi atribut-atribut baru juga merupakan salah satu cara yang berguna. Dengan splitting kita membuat atribut lebih mudah dipahami oleh model machine learning. Di lapangan sering kita temui data dengan kolom string melanggar prinsip [tidy data (Hadley Wickham)](https://www.jstatsoft.org/article/view/v059i10). 

Memisahkan sebagian data dalam kolom menjadi fitur baru memberikan keuntungan antara lain: 
1) meningkatkan performa model dengan menemukan informasi berharga
2) membuat kita lebih mudah untuk melakukan proses binning dan grouping

Ada beberapa cara untuk melakukan fungsi split, tergantung pada karakteristik kolom. Mari kita ambil salah satu contoh kasus. Sebuah atribut dengan judul “ram_hardisk” yang berisi informasi mengenai besar ram dan penyimpanan dari harddisk. Kita dapat memisahkan atribut tersebut menjadi “ram” dan “storage” untuk memudahkan model mendapatkan informasi lebih banyak dari atribut baru.

Contoh lain adalah kolom ‘full_name’. Misal kita hanya membutuhkan informasi nama hanya terdiri dari satu kata. Maka, kita dapat membagi atribut pada kolom nama dengan memisahkan data ‘first_name’ dan ‘last_name’ kemudian menggunakan salah satu atribut baru tersebut sesuai kebutuhan.

Perhatikan contoh code berikut:

~~~
import pandas as pd 
 
Developer = pd.DataFrame({'Name': ['Isyana Saraswati', 'Nicholas Saputra', 'Raisa Andriana'], 
        'Age':[30, 36, 32]}) 
print("Machine Learning Developer di Indonesia :\n", Developer) 
 
Developer[['First','Last']] = Developer.Name.str.split(expand=True) 
print("\n Split kolom 'Name', lalu tambahkan kedua kolom : \n", Developer)
~~~

Jika dieksekusi, code di atas akan memberikan hasil sebagai berikut:

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/202104161403372b7f9daa436e5047362f144644392efc.jpeg)

### Interaction feature

Dalam model machine learning, kita juga bisa menciptakan fitur atau atribut baru dari atribut-atribut yang ada. Mengkombinasikan dua atribut kadang bisa menjelaskan varian data dengan lebih baik dibanding dua atribut yang dioperasikan secara terpisah. Membuat atribut baru melalui interaksi antar atribut disebut sebagai interaction feature. Sederhananya, interaction feature adalah perkalian produk antara dua buah fitur. Analoginya adalah logika AND.

Misal kita telah memiliki sebuah atribut bernama ‘schools’ yaitu sejumlah sekolah yang berada dalam radius 5 km dari pusat kota. Kita juga memiliki atribut lain yaitu ‘accredited_A’ yang merupakan sejumlah sekolah yang telah terakreditasi A.

Dalam menentukan pilihan sekolah, kita mungkin menginginkan sekolah yang dekat tapi juga telah memiliki akreditasi A. Untuk mengakomodasi kebutuhan tersebut, kita dapat membuat atribut baru misalnya: ‘selected_schools’ = ‘schools’ x ‘accredited_A’

Interaction feature sangat mudah untuk dirumuskan, tetapi biaya komputasinya cukup tinggi. Untuk sebuah model linear dengan interaction feature berpasangan, kebutuhan waktu pelatihannya akan berubah dari O(n) menjadi O(n2), di mana n adalah jumlah fitur tunggal. 

Ada beberapa cara untuk mengatasi permasalahan ini. Pertama adalah dengan melakukan feature selection, teknik lain dalam feature engineering. Cara kedua adalah dengan menyusun (handcrafted) sejumlah kecil fitur kompleks secara hati-hati.