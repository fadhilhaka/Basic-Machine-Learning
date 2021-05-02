# Pengenalan Machine Learning

## Pengenalan Machine Learning

>“A field of study that gives computers the ability to learn without being explicitly programmed.”
-Arthur Samuel, 1959-

Istilah machine learning pertama kali dipopulerkan oleh Arthur Samuel, seorang ilmuwan komputer yang memelopori kecerdasan buatan pada tahun 1959. Menurut Arthur Samuel, machine learning adalah suatu cabang ilmu yang memberi komputer kemampuan untuk belajar tanpa diprogram secara eksplisit. 

Menurut Moroney [[1]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), prinsip atau paradigma pemrograman sejak permulaan era komputasi direpresentasikan dalam diagram berikut:

![](https://lh4.googleusercontent.com/YWmaC4c5SY05a_h7Zbvvc8h_KMeKVFHzhr2_NAKCDHbx3Oi5OrE1FS7H1VTA_HhNQjPLs7-EggWc63bfkTnjyoxzk0NxjD9XP3zp5CicDULJbd472c2h6zLEv6_Umioz4FuVntPv)

Aturan dan data adalah masukan atau input bagi sistem. Secara eksplisit, aturan diekspresikan dalam bahasa pemrograman. Tambahan masukan berupa data kemudian akan menghasilkan solusi sebagai keluaran. Paradigma pemrograman seperti pada diagram di atas sering disebut sebagai pemrograman tradisional.

Pemrograman tradisional memiliki keterbatasan.  Sifatnya rigid dengan sekumpulan aturan “if” dan “else” untuk memproses data atau menyesuaikan dengan masukan.

Sebagai contoh, kita ingin membuat sebuah program untuk mendeteksi aktivitas fisik. Kita bisa menggunakan parameter “kecepatan” sebagai data untuk membedakan aktivitas satu dan lainnya.

Misal untuk aktivitas berjalan, berlari dan bersepeda, kita membuat algoritma program dengan bahasa python sebagai berikut:

~~~
if kecepatan<4:
    aktivitas=BERJALAN
elif kecepatan<12:
    aktivitas=BERLARI
else
    aktivitas=BERSEPEDA
~~~

Kita hanya perlu mendeteksi kecepatan seseorang untuk menentukan aktivitas yang sedang dilakukannya. Tapi bagaimana jika kita diminta untuk mendeteksi aktivitas lain seperti “bermain basket”?

Kita akan menemui masalah. Orang yang sedang bermain basket akan melakukan aktivitas berjalan, kadang berlari, sekejap berhenti, dan seterusnya. Lantas, bagaimana cara program mendeteksi aktivitas tersebut? Hal ini membuat kita menyadari bahwa pemrograman tradisional memiliki keterbatasan dalam menyelesaikan masalah.

Contoh lain, misal kita ingin mengetahui bagaimana pemrograman tradisional bekerja untuk mengenali gambar. 

Pendekatan tradisional menggunakan teknik feature extraction untuk mendeteksi objek. Teknik ini merepresentasikan gambar dengan mengekstrak beberapa fitur pada gambar. Fitur adalah bagian kecil yang menarik, deskriptif, atau informatif. Ia bisa berupa sudut, tepi, skema warna, tekstur gambar, dan lain-lain.

Beberapa contoh algoritma feature extraction yang sering digunakan adalah: 
* [Harris Corner Detection](https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html)
* Scale-Invariant Feature Transform [(SIFT)](https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html)
* Speeded-Up Robust Features [(SURF)](https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html)
* Features from Accelerated Segment Test [(FAST)](https://docs.opencv.org/3.4/df/d0c/tutorial_py_fast.html)
* Binary Robust Independent Elementary Features [(BRIEF)](https://docs.opencv.org/3.4/dc/d7d/tutorial_py_brief.html)

S. Campbell dalam makalahnya yang berjudul Deep Learning vs, Traditional Computer Vision [[2]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) menyebutkan bahwa fitur-fitur diekstrak dari gambar untuk membentuk definisi dari setiap objek kelas. Dalam tahap penerapan, definisi ini dicari di berbagai tempat pada gambar. Jika sejumlah fitur ditemukan pada gambar lain, gambar diklasifikasikan sebagai gambar yang mengandung objek tertentu. 

Kesulitan yang muncul dari pendekatan tradisional ini adalah kita perlu memilih mana fitur yang penting di setiap gambar. Seiring dengan meningkatnya jumlah kelas yang akan diklasifikasikan, proses ekstraksi fitur menjadi semakin rumit. Proses ini sangat bergantung pada keahlian manusia untuk menilai fitur mana yang bisa mendeskripsikan kelas yang berbeda.

Tapi tentu ada cara yang lebih baik yaitu dengan paradigma baru pemrograman dengan machine learning.

Perhatikan diagram di bawah ini. Paradigma pemrograman pada machine learning itu berkebalikan dengan pemrograman tradisional. 

![](https://lh4.googleusercontent.com/w39xJhZzM_fAuGGHR6JNaY_gLFt-i-ANU7x08xciYK67zcGrbjXEf9QxA6bUJ82RmUWwFtrbokcA1vBaE1rCtYTChr9g4pobjnDE1uLWWy5_7lDpyG5OGACXTCFmCm-ctDkJtGqw)

Pada pemrograman tradisional kita merepresentasikan masalah menjadi aturan dalam bahasa pemrograman. Kini ketika hal itu tidak lagi memungkinkan, kita perlu mengubah alur berpikir kita dengan cara yang berbeda. Paradigma baru pemrograman dengan machine learning adalah kita memiliki banyak sekali data dan label bagi data tersebut. Kita juga telah mengetahui keterkaitan antara data dengan label sebagai suatu solusi. 

![](https://lh5.googleusercontent.com/yjLHLrfmLtDGNjpE7T81hfrbkTsjcwFtmYAQVZ0P9YMFviyzdC8aSoiAvv4G-M8z1yiGV-hVsC4sr2D2S78WNt256MBRt2uLCJiTDlJTcknQrcoPkFYuiiN7cYBchPnkFHaSBU3S)

Algoritma machine learning mencari pola tertentu dari setiap kumpulan data yang menentukan kekhasan masing-masing untuk kemudian menyimpulkan sebuah aturan. Selanjutnya, aturan ini dapat digunakan untuk melakukan identifikasi dan prediksi bagi data baru yang relevan dengan model yang kita miliki. 

![](https://lh3.googleusercontent.com/-RvAB6GO4Tl_h6LfUxnEtjTr6vKipeNdcqncjnhsx7tBjBQgnNrFY4eOlxxaDf1y_2sg8__oJNMqx3w0pG4Rbo6fGfzU_J-4OWmQ7PcxSOdyxV5R4j-5km-FS8AShgI968-52SBw)

Filter spam pada layanan email adalah contoh penerapan machine learning. Saat kita menandai satu email sebagai spam, maka program akan mempelajari anatomi email tersebut untuk mengantisipasi email-email masuk berikutnya itu spam atau bukan. Jika mirip, sebuah email baru akan masuk kategori spam, demikian juga sebaliknya. 

Setelah fase itu mulai muncul ratusan implementasi dari ML yang kita gunakan sehari-hari saat ini. Merentang dari rekomendasi video di Youtube, fitur pengenal wajah pada gambar, kontrol suara seperti pada Google Assistant, hingga sistem pemilihan pengemudi dan rekomendasi restoran pada aplikasi ojek online. Semua adalah bentuk implementasi dari machine learning. 

**Apa hubungan antara AI, ML, dan deep learning?**

Machine learning adalah cabang dari artificial intelligence. Kecerdasan buatan memiliki pengertian yang sangat luas tapi secara umum dapat dipahami sebagai komputer dengan kecerdasan layaknya manusia.

Sedangkan ML memiliki arti lebih spesifik yaitu menggunakan metode statistika untuk membuat komputer dapat mempelajari pola pada data tanpa perlu diprogram secara eksplisit.

Lebih lanjut, deep learning adalah cabang machine learning dengan algoritma jaringan syaraf tiruan yang dapat belajar dan beradaptasi terhadap sejumlah besar data. Algoritma jaringan syaraf tiruan pada deep learning terinspirasi dari struktur otak manusia.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20201229091743901b5459de9c688b2772d6ca69e37b31.jpg)

## Mengapa Machine Learning

Pemrograman tradisional memiliki keterbatasan dengan sifatnya yang rigid, untuk memproses data atau menyesuaikan dengan masukan. Menurut Muller [[3]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), penggunaan aturan untuk penentuan keputusan dengan cara ini memiliki dua kelemahan utama:

* Logika yang digunakan untuk membuat keputusan bersifat spesifik pada ranah dan masalah tertentu. Mengubah sedikit masalah, membuat kita mungkin perlu menulis keseluruhan sistem. 
* Mendesain aturan untuk sistem memerlukan pemahaman yang mendalam tentang bagaimana suatu keputusan harus dibuat oleh seorang ahli.

Padahal permasalahan terus muncul seiring waktu serta arus informasi dan data pun terus berubah. Machine learning, dengan kemampuannya yang secara alami bersifat adaptif terhadap data dan masukan baru, menawarkan solusi untuk masalah tersebut. Sebagai gambaran, berikut adalah contoh dua kategori permasalahan yang dapat diselesaikan dengan baik oleh algoritma machine learning.

### Masalah yang Solusinya Membutuhkan Banyak Penyesuaian dan Aturan

Bayangkan jika kita bertugas mengembangkan sebuah aplikasi filter spam dengan pemrograman tradisional. Langkah-langkah konvensional yang perlu kita lakukan adalah sebagai berikut:

Pertama, kita akan mendefinisikan bagaimana sebuah email termasuk kategori spam atau bukan. Misalnya, kita mengidentifikasi bahwa pada email spam umumnya terdapat kata-kata seperti “kaya”, “instan”, dan “murah”. Kemudian, kita menulis algoritma untuk setiap pola yang kita temukan pada email spam. Program pun akan menandai sebuah email spam jika menemui pola terkait. Terakhir, kita akan mengulangi kedua langkah tadi sampai program kita cukup baik untuk diluncurkan.

Karena kita menulis program menggunakan cara tradisional, hasilnya tentu daftar panjang berisi aturan-aturan rumit yang sulit untuk di-maintain, atau harus di-update secara berkala saat kita menemukan kosakata dan pola baru yang terkait dengan email spam.

Mari bandingkan jika kita menggunakan ML untuk mengembangkan filter spam tersebut. ML akan secara otomatis mempelajari pola kata-kata yang menentukan sebuah email spam atau bukan. 

Program dengan ML pun menjadi relatif lebih sederhana dan mudah untuk dipelihara. Seperti digambarkan oleh Geron [[2]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), flowchart di bawah menunjukkan bagaimana alur pengembangan sebuah proyek Machine Learning.

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20210120134955c3d0892a6b059a546ee568134dd043ab.png)

### Masalah Rumit yang Tidak Bisa Diselesaikan dengan Pemrograman Tradisional

Ada beberapa permasalahan yang sampai saat ini belum bisa dipecahkan dengan pendekatan pemrograman tradisional. Penyebabnya bisa jadi karena masalah tersebut terlalu rumit atau bisa juga karena belum diketahui algoritma pemrograman yang tepat untuk kasus tersebut. 

Misal, kita ingin membangun sebuah sistem pengenalan suara. Menurut Geron [[2]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), tahapan pertama yang perlu dilakukan adalah dengan membuat algoritma sederhana yang mampu membedakan kata “bagus” dan “indah”. Perhatikan bahwa kata “bagus” dimulai dengan suara bernada tinggi (“b”). Sebaliknya, kata “indah” dimulai dengan suara bernada rendah. Sehingga kita dapat membuat program untuk algoritma yang mengukur intensitas suara, kemudian menggunakannya untuk membedakan kedua kata tersebut. 

Pertanyaannya, dapatkah hal ini kita terapkan pada ribuan jenis kata, yang diucapkan oleh jutaan orang, dalam berbagai macam bahasa, di lingkungan yang mungkin terdistraksi oleh suara-suara lainnya? Sangat rumit, bukan? Maka solusi terbaik yang dapat kita lakukan saat ini adalah, membuat algoritma yang dapat belajar dengan sendirinya melalui banyak data berupa sampel rekaman untuk setiap kata.

### Recommendation 

Rekomendasi atau sistem rekomendasi adalah salah satu implementasi machine learning yang kita pakai hampir setiap hari. Contohnya pada saat kita belanja daring, terdapat jutaan pilihan produk pada platform tersebut. Akan membuang banyak waktu jika kita harus melihat semua opsi tersebut. Sistem rekomendasi memiliki peran penting untuk membantu kita menemukan produk yang benar-benar kita cari.

Selain pada platform belanja daring, sistem rekomendasi juga hadir dalam aplikasi yang kita pakai sehari-hari seperti Youtube dan Spotify yang merekomendasikan video dan lagu yang menarik perhatian kita. Netflix merekomendasikan film yang mungkin cocok dengan selera Anda. Twitter merekomendasikan pengguna yang mungkin ingin Anda ikuti.

## Jenis-jenis Machine Learning

### Supervised Learning

Supervised learning adalah kategori machine learning yang menyertakan solusi yang diinginkan -yang disebut label- dalam proses pembelajarannya. Dataset yang digunakan telah memiliki label dan algoritma kemudian mempelajari pola dari pasangan data dan label tersebut. Algoritma supervised learning mudah dipahami dan performa akurasinya pun mudah diukur. Supervised learning dapat dilihat sebagai sebuah mesin/robot yang belajar menjawab pertanyaan sesuai dengan jawaban yang telah disediakan manusia.

### Unsupervised Learning

Anda mungkin sudah dapat mengira bahwa pada unsupervised learning, dataset yang digunakan tidak memiliki label. Betul, model unsupervised learning melakukan proses belajar sendiri untuk melabeli atau mengelompokkan data. Unsupervised learning  dapat dilihat sebagai robot/mesin yang berusaha belajar menjawab pertanyaan secara mandiri tanpa ada jawaban yang disediakan manusia.

### Semi-supervised Learning

Ini merupakan gabungan dari supervised learning dan unsupervised learning. Di sini dataset untuk pelatihan sebagian memiliki label dan sebagian tidak.

Google Photos adalah contoh implementasi Semi-supervised Learning. Pada Google Photos kita bisa memberi tag atau label untuk setiap orang yang ada dalam sebuah foto. Alhasil, ketika kita mengunggah foto baru dengan wajah orang yang sebelumnya sudah kita beri label, Google Photos akan secara otomatis mengenali orang tersebut.

Contoh lain dari model semi supervised learning adalah Deep Belief Network (DBNs). DBNs adalah model grafis dengan multipel layer yang dapat belajar teknik mengekstrak data training secara efisien. Dua jenis layer pada DBNs adalah visible atau input layer dan hidden layer.

Menurut Geron [[4]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), DBNs berdasar pada komponen unsupervised yang disebut restricted Boltzmann machine (RBMs). RBMs dilatih secara berurutan dengan algoritma unsupervised learning, kemudian seluruh sistem disesuaikan dengan teknik supervised learning.

Campbell [[2]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) dalam tulisannya menyatakan bahwa pendekatan DBNs telah berhasil menyelesaikan pemodelan akustik pada speech recognition. DBNs menunjukkan sifat perkiraan yang kuat, peningkatan kinerja, dan merupakan parameter yang efisien.

### Reinforcement Learning

Reinforcement Learning dikenal sebagai model yang belajar menggunakan sistem reward dan penalti. Menurut Winder [[5]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), reinforcement learning adalah teknik yang mempelajari bagaimana membuat keputusan terbaik, secara berurutan, untuk memaksimalkan ukuran sukses kehidupan nyata. Entitas pembuat keputusan belajar melalui proses trial dan eror. 

Reinforcement learning memiliki empat komponen, yaitu action, agent, environment, dan reward. 

Action adalah setiap keputusan yang diambil. Misal, saat kita berkendara, action yang kita lakukan adalah mengendalikan kemudi, menginjak gas, dan mengerem.

Agent adalah entitas yang membuat keputusan, contohnya adalah perangkat lunak, atau robot, atau bahkan manusia.

Environment adalah sarana untuk berinteraksi, yang dapat menerima action dan memberikan respon berupa hasil maupun data berupa satu set observasi baru.

Reward diberikan saat agent berhasil menyelesaikan tantangan.

Mekanisme feedback ini membuat agent belajar tentang tindakan mana yang menyebabkan kesuksesan (menghasilkan reward), atau kegagalan (menghasilkan penalti). Keempat komponen ini merepresentasikan Markov decision process (MDP).

Model reinforcement learning belajar agar terus mendapatkan reward dan menghindari penalti.

[AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far), sebuah program yang dikembangkan oleh Google DeepMind adalah contoh terkenal dari reinforcement learning. AlphaGo dibuat untuk memainkan permainan Go, sebuah permainan papan kuno yang berasal dari Cina. AlphaGo mempelajari setiap langkah dalam jutaan permainan Go, untuk terus mendapatkan reward yaitu memenangkan permainan. AlphaGo terkenal setelah menjadi program komputer pertama yang berhasil mengalahkan seorang pemain Go profesional yang juga merupakan juara dunia.

## Library Populer pada Python untuk Machine Learning dan Data Science

Python merupakan bahasa paling populer yang digunakan oleh para Data Scientist dan pengembang Machine Learning (ML). Python adalah kombinasi antara general-purpose programming language yang powerful dan domain-specific scripting language yang mudah digunakan.

Salah satu faktor yang membuat Python populer adalah lengkapnya library yang dapat dipakai pada pengembangan proyek ML dari awal sampai akhir. Python memiliki library untuk data loading, visualization, statistics, data processing, natural language processing, image processing, dan lain sebagainya. 

### Numpy

[Numpy](https://numpy.org/) sangat terkenal sebagai library untuk memproses larik atau array. Fungsi-fungsi kompleks di baliknya membuat Numpy sangat tangguh dalam memproses larik multidimensi dan matriks berukuran besar. Library ML seperti TensorFlow juga menggunakan Numpy untuk memproses tensor atau sebuah larik N dimensi.

### Pandas

[Pandas](https://pandas.pydata.org/) menjadi salah satu library favorit untuk analisis dan manipulasi data. Kenapa keduanya penting? Sebelum masuk ke tahap pengembangan model, data perlu diproses dan dibersihkan. Proses ini bahkan merupakan proses yang paling banyak memakan waktu dalam pengembangan proyek ML. Library pandas membuat pemrosesan dan pembersihan data menjadi lebih mudah.

### Matplotlib

[Matplotlib](https://matplotlib.org/) adalah sebuah library untuk membuat plot atau visualisasi data dalam 2 dimensi. Matplotlib mampu menghasilkan grafik dengan kualitas tinggi. Matplotlib dapat dipakai untuk membuat plot seperti histogram, scatter plot, grafik batang, pie chart, hanya dengan beberapa baris kode. Library ini sangat ramah pengguna. 

### Scikit Learn

[Scikit Learn](https://scikit-learn.org/stable/) merupakan salah satu library ML yang sangat populer. Scikit Learn menyediakan banyak pilihan algoritma machine learning yang dapat langsung dipakai seperti klasifikasi, regresi, clustering, dimensionality reduction, dan pemrosesan data. Selain itu Scikit Learn juga dapat dipakai untuk analisis data.

### TensorFlow

[TensorFlow](https://www.tensorflow.org/) adalah framework open source untuk machine learning yang dikembangkan dan digunakan oleh Google. TensorFlow memudahkan pembuatan model ML bagi pemula maupun ahli. Ia dapat dipakai untuk deep learning, computer vision, pemrosesan bahasa alami (Natural Language Processing), serta reinforcement learning.

### PyTorch

Dikembangkan oleh Facebook, [PyTorch](https://pytorch.org/) adalah library yang dapat dipakai untuk masalah ML, computer vision, hingga pemrosesan bahasa alami. Bersaing dengan TensorFlow khususnya sebagai framework machine learning, PyTorch lebih populer di kalangan akademisi dibanding TensorFlow. Namun dalam industri, TensorFlow lebih populer karena skalabilitasnya lebih baik dibanding PyTorch.

### Keras

[Keras](https://keras.io/) adalah adalah library deep learning yang luar biasa. Salah satu faktor yang membuat keras sangat populer adalah penggunaannya yang minimalis dan simpel dalam mengembangkan deep learning. Keras dibangun di atas TensorFlow yang menjadikan Keras sebagai API dengan level lebih tinggi (Higher level API) dari TensorFlow sehingga antarmukanya lebih mudah dari TensorFlow. Keras sangat cocok untuk mengembangkan model deep learning dengan waktu yang lebih singkat atau untuk pembuatan prototipe.