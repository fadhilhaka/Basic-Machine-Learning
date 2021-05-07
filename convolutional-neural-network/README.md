## Convolutional Neural Network

Convolutional neural networks (CNNs) pertama kali dikenalkan oleh Yann LeCun et all., pada tahun 1998 dalam makalahnya “Gradient-Based Learning Applied to Document Recognition” [[22]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference). LeCun mengenalkan versi awal CNN yaitu LeNet (berasal dari nama LeCun) yang berhasil mengenali karakter tulisan tangan. Pada saat itu LeNet hanya mampu bekerja dengan baik pada gambar dengan resolusi rendah.

Database yang digunakan dalam LeCun adalah [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/index.html), terdiri dari pasangan angka 0 hingga 9 dengan labelnya. Dataset MNIST dikenal luas hingga saat ini dan banyak digunakan terutama oleh para pemula untuk melatih model machine learning.

Sejak ditemukannya LeNet, para peneliti terus melakukan riset untuk mengembangkan model CNN. Hingga pada tahun 2012, Alex Krizhevsky memperkenalkan AlexNet [[23]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), versi lebih canggih dari CNN yang memenangkan perlombaan terkenal [ImageNet](http://image-net.org/index). AlexNet ini adalah cikal bakal deep learning, salah satu cabang AI yang menggunakan multi-layer neural networks.

Selain deep learning, salah satu bidang menarik yang muncul dari perkembangan machine learning adalah computer vision. Computer Vision adalah bidang yang memberi komputer kemampuan untuk ‘melihat’ seperti manusia.

Salah satu contoh implementasi dari computer vision adalah pada pengenalan wajah, bahkan deteksi penyakit. Dan salah satu bidang yang mulai populer yaitu self driving cars.

### Bagaimana Komputer Melihat?

Komputer adalah sebuah mesin yang hanya bisa membaca angka. Sebuah gambar dalam komputer adalah matriks yang berisi nilai dari setiap pixel di gambar. Pada gambar hitam putih, gambar merupakan matriks 2 dimensi. Pada gambar berwarna, gambar merupakan matriks 2 kali 3 dimensi di mana 3 dimensi terakhir adalah jumlah kanal dari gambar berwarna yaitu RGB (red, green, blue).

![](https://lh5.googleusercontent.com/rizfJ31GoE9CCKlx0MD1KFLteFW6exlF69A6YqTX_E1aSSXl8QDwlAlqpr-pW6mkmPTti1eKJbTY6QlhcEpDi_y8qv6ezZP13493_J-9jFcplBcOgOEeUckS1VtCLw5SBzsqfJ9F)

### Klasifikasi Gambar

Salah satu peran machine learning dalam computer vision adalah pada klasifikasi gambar. Contohnya, kita punya label yaitu nama beberapa presiden Amerika Serikat. Kita ingin memprediksi siapa presiden di gambar. Jaringan saraf seperti di bawah menunjukkan probabilitas siapa presiden di dalam foto.

![](https://lh3.googleusercontent.com/opimzm9xRjlE7OvVvVOVBBcFk9DFy_2qLE18zmmtYY_iBSPTM46joY9xGpGSI09zcchhJ0DkFwx379we6Vr5zjVpoHdmFwGjRZNjbYu-mvgeLJcMkBeeeBSFNm-SzvD6SepvK-YF)

Untuk mengklasifikasikan siapa presiden dalam gambar dengan benar, model kita harus mampu untuk mengenali ciri-ciri unik yang terdapat pada wajah Lincoln, Washington, Jefferson, dan Obama.

![](https://lh6.googleusercontent.com/AeDiWJk2rUU2GZzAo8eZMhQKb2lGHWviXwFrf0ckckRAqNwO27sn1eOjREyo0CubpP3uYnz3a708rDNflcu3c6mp9JKcCBruP-eo7577PGyGu2htljFnYzP4uzVyEMWZMFqslGRC)

Setiap objek dari gambar memiliki atribut unik. Seperti pada gambar di atas, objek orang memiliki atribut unik berupa hidung, mata, dan mulut. Pada objek mobil terdapat atribut roda, lampu, dan plat nomor, sedangkan objek rumah memiliki atribut pintu, jendela, dan tangga.

### Convolutional Layer

Sebuah jaringan saraf biasa mengenali gambar berdasarkan piksel-piksel yang terdapat pada gambar. Teknik yang lebih optimal adalah dengan menggunakan convolutional layer di mana alih alih mengenali objek berdasarkan piksel-piksel, jaringan saraf dapat mengenali objek berdasarkan atribut-atribut yang memiliki lebih banyak informasi.

Convolutional layer berfungsi untuk mengenali atribut-atribut unik pada sebuah objek.  Atribut-atribut yang lebih rendah membentuk atribut lebih tinggi contohnya atribut wajah dibentuk dari atribut mata, telinga, dan hidung. Atribut mata dibentuk dari garis, lengkungan dan bintik hitam.

![](https://lh5.googleusercontent.com/6xbwWtqDHNNdNysh4d_C3pkfJlKlHBy2ysqbWlGB5g49LvIe1_WeetBnxVZdca1_aXDGqBZTgiwEGUuxlKReCY6-sZnjEQJVuBnuj5niDgKCVJjvmNqa4HWQL7depQb1e4r_R3ik)

**Filter**

Convolutional layer dapat mengenali atribut pada objek menggunakan filter. Filter hanyalah sebuah matriks yang berisi angka-angka. Pada gambar di bawah terdapat 3 buah filter masing-masing merupakan matriks 3x3 dan sebuah objek yaitu gambar berisi huruf X. Filter yang berada di sebelah kiri digambarkan dapat mengenali garis yang terdapat pada kotak hijau. Setiap filter berbeda dapat mengenali atribut yang berbeda seperti, filter di kanan dapat mengenali atribut objek x yang berada di kotak merah.

![](https://lh3.googleusercontent.com/r0g6lfdDX3Ampe4p9pO4ugY9HIjiMTVwX6VO6UwR06cgdTq4cF4uIyA7fUz3r_vhIfHg4rH-72vkgO1amvJrdN0z86patxd9OJVQR5WNuVHv0F3y7aM2g0ch4EM9c3hxE5uitCGC)

Contoh lain dari filter dapat Anda lihat di bawah. Pada sebuah gambar perempuan, aplikasi dari filter yang berbeda menghasilkan gambar yang berbeda.

Kita dapat membedakan seekor kuda dan manusia berdasarkan bentuknya bukan? Nah, dengan filter seperti pada gambar yang paling kanan, kita dapat mendeteksi garis-garis yang bisa menunjukkan apakah seseorang merupakan kuda atau manusia berdasarkan bentuk garisnya.

![](https://lh6.googleusercontent.com/YfjVAMjeoINAkjDvif5667zAeBOcbS15oL5I_wn2Y9_RMw7sIcSr4z3LcSjza8XHUO22pTbWV63AdMTdVZ5gYw3-o-1HmX5MPuZ7gm_HtKQvFnrNf5tSvqL-TfUK-XPUWp4Dm1Pc)

**Proses Konvolusi**

Proses konvolusi adalah proses yang mengaplikasikan filter pada gambar. Pada proses konvolusi ada perkalian matriks terhadap filter dan area pada gambar. Pada ilustrasi di bawah terdapat sebuah gambar, filter, dan hasil dari proses konvolusi terhadap gambar. Animasi selanjutnya menunjukkan bagaimana proses konvolusi pada ilustrasi sebelumnya dikerjakan.

![](https://lh6.googleusercontent.com/Ls68A5L7pim9juer-pOpVKLdjgx9OUDyMRANi_L_YYKQ-ApkIQR94WG0Lue80ZeNmpqzYdxCE19zbfS7KC0xHli3D-k3CiHNPufN_UbIYnxuC7q9blBnRLJj9PvcHUlDvZ1chOKF)

![](https://lh5.googleusercontent.com/LuUlZsYRviKLl69CymdN8fd51RUT-4e0AaFyO8Yfcn15lV32A2ot1vHgbeESGp2Kgv1vOpoItthkygOjyTOxvh3z5nWTPd96z0oGdbffvlwIM-vU6N940m2148SSfHnz0TWG-9-P)

Ketika proses konvolusi selesai, hasil dari konvolusi tersebut dapat dijadikan masukan untuk sebuah MLP.

**Max Pooling**

Pada sebuah jaringan saraf tiruan, umumnya setelah proses konvolusi pada gambar masukan, akan dilakukan proses pooling. Pooling adalah proses untuk mengurangi resolusi gambar dengan tetap mempertahankan informasi pada gambar. Contohnya seperti pada gambar berikut di mana ketika resolusi dikurangi sampai batas tertentu kita masih bisa mendapatkan informasi mengenai objek pada gambar.

![](https://lh6.googleusercontent.com/yomwcQahSG5PL2eFnBWkbj07P0wOYVF-r-Eh_TC41v0LSpIHUAmXrPvAIDYv7uK-3QTMb6FNwqNlP_lPsRH4nfsNpBwsc-QCYvvke5hDHvNSYVX-FUwllx2Et8QKNlez_NjABWT8)

Salah satu contoh dari pooling adalah max pooling. Pada max pooling di antara setiap area dengan luas piksel tertentu, akan diambil satu buah piksel dengan nilai tertinggi. Hasilnya akan menjadi gambar baru. Animasi di bawah menunjukkan contoh max pooling dengan ukuran 2x2 piksel pada gambar berukuran 4x4 piksel. Hasil dari max pooling adalah gambar dengan ukuran 2x2 piksel.

![](https://lh3.googleusercontent.com/VkXlxw5LBIedeym8qc3W31nvCUuAih00XDXX34s_asH_APlyKeFjSI9EdSoBEe0g-kWqbDrRBg9qREOfggQBtLEAc8yHwdIOfmK7_Xm-eHjrZDbSJ8R-RsiTYqbu7o-Td5BqA1rC)

Proses max pooling dipakai karena pada praktiknya, jumlah filter yang digunakan pada proses konvolusi berjumlah banyak. Ketika kita menggunakan 64 filter pada konvolusi maka akan menghasilkan 64 gambar baru. Max pooling membantu mengurangi ukuran dari setiap gambar dari proses konvolusi.

**Arsitektur Convolutional Neural Network**

Arsitektur CNN adalah sebuah jaringan saraf yang menggunakan sebuah layar konvolusi dan max pooling. Pada arsitektur CNN di bawah, sebuah gambar masukan dideteksi atribut/fitur nya dengan menggunakan konvolusi 3 filter. Lalu setelah proses konvolusi akan dilakukan max pooling yang menghasilkan 3 buah gambar hasil konvolusi yang memiliki resolusi lebih kecil. Terakhir, hasil max pooling dapat dimasukkan ke dalam sebuah hidden layer MLP.

![](https://lh5.googleusercontent.com/1CjZizogbIO6MyF9G5taJTJzrei16WC0Gy5hw69O4xUz_yihq0KpzFmfBfRNl4HKv9JrNLgq4ynbn9uMVkh9TnDONDkpklBe5qwkB1HlciJdV25xLX3dt1-_IRDofAJZ1OPiwW-g)

Kita juga dapat menggunakan beberapa lapis konvolusi dan max pooling sebelum mulai memasukkannya ke hidden layer sebuah MLP. Cara kerjanya sederhana. Kita bisa melakukan proses konvolusi dan max pooling setelah lapisan max pooling sebelumnya. Pada contoh di bawah terdapat 2 kali proses konvolusi dan max pooling sebelum hasilnya dimasukkan ke dalam hidden layer.

![](https://lh6.googleusercontent.com/zcl742abAlEz494zUHwz7MBrTD-iaPoCPP5UTnFp9yTA2n8vhR_JcUrqcu8Nea1Rmr-WRP_CCCRGBdS-zNgK2dh3RpeMybRpUHv3IkTZoR6yA5-WV7o3rO_qFPT8E0FW-11wQ-rT)

Dengan beberapa lapis proses konvolusi, makin detail fitur yang dapat dikenali dari gambar. Contohnya pada proses konvolusi pertama dapat mendeteksi wajah dari seorang manusia. Lalu pada proses konvolusi kedua, wajah hasil konvolusi pertama dapat dideteksi fitur yang lebih detail seperti hidung, mata, dan telinganya sehingga, model makin pintar membedakan wajah setiap orang.

[Convolutional Neural Network](https://cs231n.github.io/convolutional-networks/)