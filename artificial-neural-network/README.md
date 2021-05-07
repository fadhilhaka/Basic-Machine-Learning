# Neural Network

## Artificial Neural Network

Para penemu kerap mencari inspirasi dari alam dan fenomena sehari-hari. Dulu pada tahun 1904 burung menginspirasi Wright bersaudara untuk menciptakan pesawat laik terbang perdana di dunia. Kini, desain futuristik kapal selam tenaga nuklir milik Royal Navy Inggris, contohnya, juga meniru bentuk paus, belut, dan ikan pari manta.

Demikian halnya dengan machine learning dan jaringan saraf tiruannya. Jaringan Saraf Tiruan atau Artificial Neural Network (ANN) adalah sebuah model machine learning yang terinspirasi dari neuron/saraf yang terdapat pada otak manusia.

ANN merupakan salah satu model ML yang multiguna, powerful, dan memiliki skalabilitas tinggi. Dengan kelebihan tersebut ANN sangat ideal dipakai dalam menangani masalah ML yang sangat kompleks seperti mengklasifikasi miliaran gambar, mengenali ratusan bahasa dunia, merekomendasikan video ke ratusan juta pengguna, sampai belajar mengalahkan juara dunia permainan papan GO.

### Neuron dalam Otak

Sebelum kita belajar mengenai saraf tiruan, kita akan mengenal lebih dahulu saraf biologis (neuron). National Institute of Neurological Disorders and Stroke [[18]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) dalam tulisannya yang berjudul “Brain Basics: The Life and Death of a Neuron” menyatakan bahwa neuron atau saraf adalah pembawa pesan/informasi. Mereka menggunakan impuls listrik dan sinyal kimiawi untuk mengirimkan informasi antara area otak yang berbeda, serta antara otak dan seluruh sistem saraf.

Sebuah saraf terdiri dari 3 (tiga) bagian utama, yaitu badan sel yang disebut nukleus, akson, dan dendrit. Nukleus berisi materi genetik dan bertugas mengontrol seluruh aktivitas sel. Akson adalah cabang yang terlihat seperti ekor yang panjang, bertugas mengirimkan pesan dari sel. Panjang akson berkisar antara beberapa kali lebih panjang dari badan sel, sampai 10 ribu kali lebih panjang dari badan sel. Sedangkan dendrit adalah cabang-cabang pendek yang terlihat seperti cabang pohon, bertugas menerima pesan untuk sel.

Setiap ujung akson dari sebuah neuron terhubung dengan dendrit dari neuron lainnya. Neuron berkomunikasi satu sama lain dengan mengirimkan senyawa kimia yang disebut neurotransmitter, melintasi ruang kecil (synapse) antara akson dan dendrit neuron yang berdekatan. Ketika sebuah neuron mendapatkan rangsangan, neuron tersebut akan mengirim sinyal ke neuron lainnya. Seperti ketika kita tidak sengaja menyentuh panci yang panas, saraf di  tangan kita mengirim sinyal ke saraf lain sampai ke otak dan kita merespon dengan cepat. Pengiriman sinyal antar neuron terjadi sangat cepat yaitu hanya dalam beberapa milidetik.

![](https://lh6.googleusercontent.com/_1IwNKpvuCdHWvk8kwhpO8hvj4lSOwnrjk1jePPFe6-cg35mO3PjOXDx0AEVM5T8X2-ffc9sxZbe5faomTTABJZG7sWPnFBo4wf52LV1AIuDekV-zrRZp3EpNAsgLn0xml_3TiBd)

Cara kerja dari sebuah neuron sangatlah sederhana. Namun, neuron-neuron tersebut terorganisir dalam sebuah jaringan berisi miliaran neuron. Setiap neuron lalu terhubung dengan beberapa ribu neuron lainnya. Dengan jumlah yang luar biasa besar tersebut, banyak pekerjaan kompleks yang dapat diselesaikan.

### Perceptron

Perceptron adalah komponen dasar pembangun jaringan saraf tiruan. Frank Rosenblatt dari Cornell Aeronautical Library adalah ilmuwan yang pertama kali menemukan perceptron pada tahun 1957 [[19]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference). Perceptron pada jaringan saraf tiruan terinspirasi dari neuron pada jaringan saraf di otak manusia. Pada jaringan saraf tiruan, perceptron dan neuron merujuk pada hal yang sama.

Sebuah perceptron menerima masukan berupa bilangan numerik. Perceptron kemudian memproses masukan tersebut untuk menghasilkan sebuah keluaran. Agar lebih memahami cara kerja perceptron, kita akan menggunakan diagram di bawah.

![](https://lh3.googleusercontent.com/pDeBAcup_I82X7CsRjSjHRxsPtUFLQnXjV8JKPRTZykkvRZpYeql-w2IwP0gkuJNHP4XkKc0Il-pB5g9LS1rvPeMStF3Ypj51eiNdzai60GgQnbKwj61Rw3WxFPb8p9vrGhkHaqq)

Sebuah perceptron terdiri dari 5 komponen yaitu:

1. Input (xi)
2. Bobot atau weights (Wi) dan bias (W0)
3. Penjumlahan atau sum (∑)
4. Fungsi aktivasi atau non linearity function (⎰)
5. Output (y)

Berikut adalah proses yang menjelaskan bagaimana perceptron bekerja, yaitu:

* Input menerima masukan berupa angka-angka. 
* Setiap input memiliki bobot masing-masing. Bobot adalah parameter yang akan dipelajari oleh sebuah perceptron dan menunjukkan kekuatan node tertentu. 
* Selanjutnya adalah tahap penjumlahan input. Pada tahap ini setiap input akan dikalikan dengan bobotnya masing masing, lalu hasilnya akan ditambahkan dengan bias yang merupakan sebuah konstanta atau angka. Nilai bias memungkinkan Anda untuk mengubah kurva fungsi aktivasi ke atas atau ke bawah sehingga bisa lebih fleksibel dalam meminimalisasi eror. Penjelasan lebih lanjut tentang bias dapat Anda pelajari pada [tautan](https://towardsdatascience.com/why-we-need-bias-in-neural-networks-db8f7e07cb98) berikut. Hasil penjumlahan pada tahap ini biasanya disebut weighted sum.
* Aplikasikan weighted sum pada fungsi aktivasi atau disebut juga Non-Linearity Function. Fungsi aktivasi digunakan untuk memetakan nilai yang dihasilkan menjadi nilai yang diperlukan, misalnya antara (0, 1) atau (-1, 1). Fungsi ini memungkinkan perseptron dapat menyesuaikan pola untuk data yang non linier. Penjelasan lebih lanjut tentang fungsi aktivasi akan diulas pada paragraf di bawah.
* Setelah semua langkah di atas, akhirnya kita memperoleh output, yaitu hasil dari perhitungan sebuah perceptron yang merupakan bilangan numerik.

Fungsi matematis dari perceptron dapat kita lihat di bawah. Rumus di bawah merupakan notasi matematis yang menjelaskan proses yang kita bahas sebelumnya. Keluaran (ŷ) dari perceptron merupakan bias (W0), ditambah dengan jumlah setiap input (Xi) yang dikali dengan bobot masing-masing (Wi) sehingga menghasilkan weighted sum, kemudian dimasukkan ke dalam fungsi aktivasi (g).

![](https://lh5.googleusercontent.com/68N1mGs6v0nEfRI2ZWAEHys6Xd6Qmez-J_Bb-8wn3l7npMCaup6lV9-yytQeLGBrlZFmc5LLhKon5oYJ_8tbXTn_D9QbWfynZk3eYSI9227cShRmbmnIo30HqavjUvH0_vb0O-Hc)

Fungsi aktivasi pada perceptron bertugas untuk membuat jaringan saraf mampu menyesuaikan pola pada data non linier. Seperti yang sudah pernah dibahas sebelumnya, mayoritas data yang terdapat di dunia nyata adalah data non linier seperti di bawah.

![](https://lh4.googleusercontent.com/PmpW5B0TqjSHCl4VAIwy2OkRKEr0fQSwxTRk6zpaNBPRwtw59VdVvanh3Kv7eZiUvSmd3Q5urNsvoiIQ5pfh9kdcEYa_-k2lRM0Bn0J_9t1Y0iLm-FSQDcqqB8OQNsNo7FmoWD-I)

Fungsi aktivasi lah yang memungkinkan jaringan saraf dapat mengenali pola non-linier seperti di bawah. Tanpa fungsi aktivasi, jaringan saraf hanya bisa mengenali pola linier seperti garis pada regresi linier.

![](https://lh3.googleusercontent.com/YbeB93g4yLpVwCw8hv-b19MjC7dvrQMlRzRaq5BqCa5vlytbjXquyuQ0RRr2JdRS798pGOtFEOu090SlICgKBwgot4lIk802w0UgwY95xL_CrZzn-hxZy4B0zpbYC8oVDKcq_i36)

Ada 3 fungsi aktivasi yang paling umum yaitu sigmoid function, hyperbolic tangent, dan rectified linear unit (ReLU).

**Sigmoid atau Logistic Function**

Fungsi ini berada di antara nilai 0 hingga 1 sehingga biasanya digunakan untuk memprediksi model probabilitas yang outputnya ada di kisaran 0 dan 1. Dengan kemiringan yang halus (smooth gradient) memungkinkan Gradient Descent (algoritma pengoptimalan yang mampu menemukan solusi optimal untuk berbagai masalah) berprogres pada setiap langkahnya. 

Selain itu, fungsi sigmoid memberikan nilai prediksi yang lebih jelas. Dari gambar di bawah [[20]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), perhatikan bahwa nilai X di bawah -2 atau di atas 2 cenderung memberikan nilai prediksi yang sangat dekat dengan 0 atau 1.

![](https://lh5.googleusercontent.com/xEJkmEAh7n96PZyOWJXDOmojkW2QCvmstP2VxLgenpeYKReLC2TF-jfKbyCiEMwPkwOXKOP_qEkp7gSbg0JH5q-rRHvMXDpiHSV8vCEiVtbn3slw3qOOVqn_PrpuyDXtzol-h2ah)

**Hyperbolic Tangent (tanh)**

Sama seperti fungsi sigmoid, fungsi tanh berbentuk S, kontinu, dan dapat dibedakan. Perbedaannya adalah nilai keluarannya berkisar dari -1 hingga 1. Rentang tersebut cenderung membuat keluaran setiap lapisan kurang lebih berpusat di sekitar 0 pada awal pelatihan sehingga dapat membantu mempercepat konvergensi [[4]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference). Sifat zero centered ini juga membuat fungsi ini lebih mudah dalam memodelkan masukan yang memiliki nilai sangat negatif, netral, dan sangat positif.

![](https://lh6.googleusercontent.com/K4rbhP9KhJcdZW4vvwngFYRtBMvVagDhbRNidm6oxuXAuXukR8GqyNneKg16pNubNt1N35Cfu4-2G2lowjDN5x60jfgLrHdjQvA4pG9J5KtrrxLyfF0g0Z1IyzkXBgHVwoo2DIcL)

**Rectified Linear Unit (ReLU)**

Fungsi ReLU bersifat kontinu meski kemiringannya berubah secara tiba-tiba dan nilai turunannya bernilai 0 pada z < 0. Akan tetapi, fungsi ini bekerja dengan sangat baik dan membuat jaringan bekerja secara efisien sehingga mempercepat waktu komputasi. Karena hal inilah, fungsi aktivasi ini sering digunakan sebagai fungsi aktivasi default pada jaringan saraf tiruan/ANN. 

![](https://lh3.googleusercontent.com/xdFU1XEjVVCgR-hNWv4gSEpFc4FYUER2rSkZX-ayAQj_7HSf1vcxi8nunuucKOONJAJtvpN_W9H7Y8H6Wn6NxfSeFciOkixJSuCgwUb4N2vgzhqo2JFYatRU9Fa5-fhuw1OP15bN)

Beberapa fungsi aktivasi dan turunannya ditunjukkan pada gambar berikut:

![](https://lh4.googleusercontent.com/_nu8Wv3_k309tRK17nEDdiaAF07cXMeZe2Boju_xH-t_lHp8KbEZPtnjjWp75wAl0p_1O1q_rwCfELefTV7qls8rKxFSbdKl0fjgKl9eHPd--3HvWmKBVVsDJUG3QhtqP3Cg6i6e)

[Fungsi aktivasi](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/)

Selanjutnya mari kita kembali pada pembahasan mengenai perceptron. Kita akan menggunakan diagram yang lebih sederhana untuk mengilustrasikan perceptron. Pada diagram di bawah, x1, x2, dan x3 adalah input, dan z adalah perceptron. Bias dan bobot tidak masuk dalam ilustrasi agar diagram terlihat lebih sederhana.

![](https://lh4.googleusercontent.com/eqMZ4et4tu7GtSemzoA0egg2LdPEyaetG3B0EOSs9bl33X_3O5DNcS6Ymlu15U9HiAfmcywngby4okeswTenjYOWDztT2GgCMSODEFtqMVyUPLAYXJ7M4Vv4ZJ7BMiZFfC8Mihfa)

Pada gambar di bawah terdapat 2 buah perceptron yaitu Z1 dan Z2. Karena koneksi tiap perceptron dan tiap input sangat padat, jadi dinamakan dengan dense layer. Dense layer adalah sebuah lapisan yang terdiri dari 2 perceptron atau lebih.

![](https://lh6.googleusercontent.com/3PNB9pQ0JrJl6O9rzm-hz73Qx0a_n1VwN0YJ2Nm4lppcKjN1LlddW7n0cajKCMhxMjG1sp3_-ODyXKq32BmAFO8XtNxoJ5kgHAMaSj9IPxs-4krFbgir4oUFybMn884P1Ih49-GC)

Selanjutnya kita akan membahas tentang hidden layer. Sebuah hidden layer adalah dense layer yang berada di antara input layer dan output layer. Pada ilustrasi di bawah, jaringan saraf di sebelah kiri memiliki 1 hidden layer dan jaringan saraf di sebelah kanan memiliki 4 buah hidden layer.

![](https://lh6.googleusercontent.com/5BxtlIv7b2QvWMR3EMVS24DtJwk8UGAkpk9XrleX_IPh0k3ZxpyMTTcv7AHFHkllN77TktS0VPPXBA2fDandkoNtvLYc3ZiySgKiO3ppYEYD8y9Q1qoUR81DNPD9_vpXSIsxweXW)

Dalam sebuah jaringan saraf tiruan, input layer dan output layer harus selalu ada, namun untuk hidden layer bisa ada beberapa atau tidak sama sekali. Hidden layer dan output layer sama-sama merupakan sebuah layer yang memiliki beberapa perceptron. Sedangkan input layer adalah sebuah layer yang hanya menampung angka-angka.

Hidden layer diberikan nama hidden karena sifatnya yang tersembunyi. Pada sebuah sistem jaringan saraf, input dan output layer merupakan lapisan yang dapat kita amati, sementara hidden layer, tidak.

Pada sebuah jaringan saraf tiruan, semakin banyak jumlah hidden layer dalam sistem, semakin lama jaringan saraf tersebut memproduksi hasil, namun juga semakin kompleks masalah yang dapat diselesaikan.