# Unsupervised Learning

Setelah kita memberikan sekumpulan data tanpa label, model machine learning akan mempelajari pola dan struktur pada data berdasarkan hubungan atau keterkaitan antar variabel pada data. Model kemudian akan mengelompokkan data ini ke dalam beberapa klaster yang berbeda. Teknik ini disebut sebagai clustering. 

Contoh kasus untuk teknik clustering adalah customer segmentation. Dari data ribuan pengunjung sebuah website ecommerce, model akan belajar sendiri untuk mengelompokkan pengunjung. Bisa berdasarkan waktu kunjungan, lama kunjungan, penggunaan fitur search, jumlah klik, dan sebagainya.

Model unsupervised learning akan menentukan segmen market dan mengelompokkan pengunjung  ke dalam segmen market yang berbeda. Dengan output dari model ini, pengelola ecommerce dapat menentukan strategi untuk meningkatkan penjualan atau strategi lain yang dirasa perlu diambil untuk kelanjutan bisnis.

Metode unsupervised learning yang sekarang sedang sangat populer adalah generative adversarial networks (GANs).

Terinspirasi dari teori game, GAN bekerja dengan cara membuat dua jaringan syaraf tiruan berkompetisi. Lapan [[7]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference) dalam bukunya menyatakan, ketika kita memiliki dua jaringan syaraf yang bersaing, jaringan pertama mencoba menghasilkan data palsu untuk mengelabui jaringan kedua, sedangkan jaringan kedua mencoba untuk membedakan data palsu tersebut dari data sampel kumpulan data kita. Seiring waktu, kedua jaringan menjadi semakin ahli dalam tugas-tugas mereka dengan menangkap pola spesifik yang halus dalam kumpulan data.

Beberapa algoritma unsupervised learning yang penting untuk Anda ketahui adalah: 
* Clustering
* Dimensionality reduction
* Anomaly detection
* Density estimation

## Clustering

Klaster (cluster) adalah sebuah grup yang memiliki kemiripan tertentu. Pengklasteran adalah sebuah metode machine learning untuk mengelompokkan objek-objek yang memiliki kemiripan, ke dalam sebuah klaster. Karena termasuk kategori unsupervised, maka dataset yang digunakan model clustering tidak memiliki label.

Menurut Andriy Burkov dalam buku The Hundred Page Machine Learning Book [[8]](https://github.com/fadhilhaka/Basic-Machine-Learning/tree/main/reference), clustering atau pengklasteran adalah sebuah metode untuk memberi label pada data tanpa bantuan manusia.

Contohnya adalah ketika kita memiliki data pengunjung web toko online kita seperti gambar di bawah. Kemudian kita ingin mengembangkan sebuah model yang bisa mengelompokkan pengunjung yang memiliki kemiripan. 

Misalnya, diketahui bahwa 80% pengunjung toko online Anda adalah perempuan, sementara 20% nya adalah laki-laki. 60% dari pengunjung perempuan mengunjungi toko online Anda pada hari kerja, sementara sisanya berkunjung pada akhir minggu. 

Contoh lain, 40% pengunjung toko online Anda berasal dari Pulau Jawa, 55% berasal dari pulau lain di seluruh Indonesia, dan 5% sisanya berasal dari luar negeri.

Tujuan pengelompokkan kemiripan ini adalah agar kita mengetahui target market yang sesuai untuk setiap kelompok.

Sebuah model pengklasteran akan membandingkan atribut setiap pengunjung lalu membuat sebuah klaster yang diisi oleh pengunjung yang memiliki kemiripan karakteristik/atribut yang tinggi.

![](https://lh6.googleusercontent.com/P7b4PxAbbO3ycT9JKg8QG9SmT9cvwdYdHmZ4ltxU5TM-Gjh_rpQjZ1LQbnxm3f8Lozes-TrqpVKDPNQurvSctvlwlbSHp7CUYGcB_22SAV9dastTPLNK1SSD_6UydZIxbNuGUdnz)

Contoh di atas dikenal juga sebagai customer segmentation, salah satu kasus yang populer di industri, di mana bisnis mengelompokkan pelanggan agar bisa memberikan penawaran yang sesuai untuk setiap kelompok. Misal, kelompok pengunjung wanita dengan rentang usia 25 sampai 35 tahun tentu akan memiliki selera yang berbeda dengan pengunjung wanita pada rentang usia 40 tahun ke atas. Customer segmentation ini penting agar setiap target kelompok mendapatkan penawaran yang sesuai sehingga dapat memberikan kontribusi positif terhadap revenue toko.

## K-Means Clustering

Pengklasteran K-Means adalah sebuah metode yang dikembangkan oleh Stuart Lloyd dari Bell Labs pada tahun 1957. Lloyd menggunakan metode ini untuk mengubah sinyal analog menjadi sinyal digital. Proses pengubahan sinyal ini juga dikenal sebagai Pulse Code Modulation. 

Pada awalnya metode K-means hanya dipakai untuk internal perusahaan. Metode ini baru dipublikasikan sebagai jurnal ilmiah pada tahun 1982. Pada tahun 1965, Edward W. Forgy mempublikasikan metode yang sama dengan K-Means sehingga K-Means juga dikenal sebagai metode Lloyd-Forgy.

Untuk melihat bagaimana K-Means bekerja, kita akan menggunakan ilustrasi dari tulisan [M. Benbihi](https://www.astateofdata.com/machine-learning/k-means-cluster-k-means-clustering-how-it-works/). Perhatikan bahwa data yang kita gunakan terdapat 12 sampel dan data ini merupakan data 1 dimensi.

![](https://lh4.googleusercontent.com/be5__TLtf8KhUTKbLutm3t-fp3r6lo8VieRI626CpATV4Lrpd0bsCClFKwvzLdDWzGMeafxTJn2jCJdhu4KsRiYjbKZzekAVmWPVjjAi0DTXvXg0DTj0CKhFSVkEg7Z0iWGQvyAH)

Hal yang paling pertama K-Means lakukan adalah memilih sebuah sampel secara acak untuk dijadikan centroid. Centroid adalah sebuah sampel pada data yang menjadi pusat dari sebuah klaster. Kita bisa melihat pada gambar bahwa 3 sampel yang dijadikan centroid diberi warna biru, hijau dan kuning.

![](https://lh4.googleusercontent.com/PcS-np0CdgCoeQCjenA7oS3endrLKXR0OzLmzU7VmfKLtn2viiyOnjD0S-ldq9eWvhFGPp0lLu6W7-4k_WDOIe7Dxir2VUzxSinjfNg4wj2jtr0Y2Vpw-JlJU2eTyzrkuv1c-2JZ)

Kedua, karena centroid adalah pusat dari sebuah klaster, setiap sampel akan masuk ke dalam klaster. Ini bermula dari centroid terdekat dengan sampel tersebut. Pada contoh di bawah, sampel yang ditunjuk anak panah memiliki jarak terdekat dengan centroid warna hijau. Alhasil, sampel tersebut masuk ke dalam klaster hijau.

![](https://lh4.googleusercontent.com/iOfltf4GLLUsfu_RUn1x7uNB0BXBzyUmvxPsZgHBwpJVd-9NzFxC1WwmyHEVcK1rAk_oVY9-DTvJSoe7atTpB18CuExx_MliCZpDv4DTGl5yuoPPgxGna9ZbMzllxLDsqLzS29z3)

Berikut adalah hasil ketika tahap kedua selesai.

![](https://lh5.googleusercontent.com/zRSDWg4GPIzXowq8qkPagwOMleTFDBark9sJbngr1ggO-tE3fNUH2mT6cFDbopqPum1S1RdwTH_sRGV2ZWkOu9pSsp6fk1U8_wA-XcH0nykbmoqVv079fNdxIRV1fukLbkxglmpA)

Ketiga, setelah setiap sampel dimasukkan pada klaster dari centroid terdekat, K-Means akan menghitung rata-rata dari setiap sampel dan menjadikan rata-rata tersebut sebagai centroid baru. Rata-rata di sini adalah titik tengah dari setiap sampel pada sebuah klaster. Pada gambar dibawah, rata-rata yang menjadi centroid baru digambarkan sebagai garis tegak lurus.

![](https://lh3.googleusercontent.com/xy0D6H_5uXhrIL0qTmDYRYLgtrEA4tlb5K-YK7mzJCJD2VShHE9CDV67iQylf606kj3mFWNk0aZ6sCVEfhK1DcJiUnNv2ikZAElvLu3ekpNmvgUlW29-cmS0Pa8n7h7PbI1xpk7r)

Keempat, langkah kedua diulang kembali. Sampel akan dimasukkan ke dalam klaster dari centroid baru yang paling dekat dengan sampel tersebut.

![](https://lh6.googleusercontent.com/rmpskvblSaDTYhL2cbRTWAmoo0MUyUWpeCgQDKegCu-kCtdm2oNMwJcYEezw0VFwRPntD-pIMNAut9aThkS-nqnDlDo5wksWSdHyKwUgwEQVybGNhyyFh_MFBQ-apt7PDBlhBekO)

Pada tahap ini Anda mengulangi langkah ketiga, yaitu menemukan rata-rata dari klaster terbaru. Anda akan menemukan rata-rata tiap klaster di tahap keempat akan sama dengan rata-rata tiap klaster pada tahap ketiga sehingga centroidnya tidak berubah. Ketika centroid baru tidak ditemukan, maka proses clustering berhenti.

Untuk mengukur kualitas dari pengklasteran, K-Means akan melakukan iterasi lagi dan mengulangi lagi tahap pertama yaitu memilih sampel secara acak untuk dijadikan centroid. Gambar di bawah menunjukkan K-Means pada iterasi kedua mengulangi kembali langkah pertama yaitu memilih centroid secara acak.

![](https://lh3.googleusercontent.com/lA8YFNc0isWbiqeePo--qrZJ2E11OQBsNFeLyq0MoZUGHyKSBP15bc3V3lhd5cWN_ZI33q-dqCFMOTF7ClQRnHGIxSxen4HVITuhMqmqnpyOCaYiA72GDbJSGjIl6GuZ5yZYz1sz)

Untuk iterasi kedua, Anda bisa mempraktekkan langkah yang sudah dijelaskan sebelumnya untuk menguji pemahaman Anda.Hasil dari iterasi kedua adalah sebagai berikut.

![](https://lh3.googleusercontent.com/StQn_-TgNE9tXxD-lU8gMsFyElizpD1nTU9SWcPcn2_pkK05zYIVuJ27AwSPd-13Zn0ASW0aFfCgtxaDzDNSfoqBM1PemjVYIqx2_kzoTXEY11y09tzKZsjWn7HEI5dE90qD9-Js)

Hasil dari iterasi kedua terlihat lebih baik dibanding iterasi pertama. Untuk membandingkan klaster setiap iterasi, K-Means akan menghitung variance dari tiap iterasi. Variance adalah persentase jumlah sampel pada tiap klaster. Gambar di bawah menunjukkan variance pada iterasi pertama.

![](https://lh5.googleusercontent.com/tkjoNeRqAWfOR9balYTfe5FCB0zz3ZfVFohzAR__ydSycYhI-UNMIGxUBPsWtV2ZQZ8OAfLb-WQ1rEvNRoHxdzWxqQ06VXnLeiflMk99aoG4ON3XHM8Y6Kg3A1ymAWawbYP11kuq)

Kita bisa melihat bahwa di iterasi kedua, variance nya lebih seimbang dan tidak condong pada klaster tertentu. Sehingga, hasil dari klaster iterasi kedua lebih baik dari iterasi pertama. Jumlah iterasi dari K-Means ditentukan oleh programmer, dan K-Means akan berhenti melakukan iterasi sampai batas yang telah ditentukan.

Untuk data yang memiliki 2 dimensi atau lebih, k-means bekerja dengan sama yaitu menentukan centroid secara acak, lalu memindahkan centroid sampai posisi centroid tidak berubah. Animasi di bawah akan membantu Anda untuk melihat bagaimana K-Means bekerja pada data 2 dimensi.

![](https://lh3.googleusercontent.com/R7DC4KEuS9Lnv0457zCHYPxzDiR-IhFv6XlhEEl4kFtH7UpfOVEnefgpy_IsHoipW62I6idy_-8a1B0RA_fZ8oGcK_PQ470NgDC8FHc-_bPAn9tql8gh1pgbIFdaXi6_Vqz2moip)

### Metode Elbow

Cara paling mudah untuk menentukan jumlah K atau klaster pada K-means adalah dengan melihat langsung persebaran data. Otak kita bisa mengelompokkan data-data yang berdekatan dengan sangat cepat. Tetapi cara ini hanya bekerja dengan baik pada data yang sangat sederhana.

Ketika masalah clustering lebih kompleks kita bisa menggunakan metode Elbow.

Ide mendasar dari metode elbow adalah untuk menjalankan K-Means pada dataset dengan nilai K pada jarak tertentu (1,2,3, .., N). Kemudian hitung inersia pada setiap nilai K. Inersia memberi tahu seberapa jauh jarak setiap sampel pada sebuah klaster. Semakin kecil inersia maka semakin baik karena jarak setiap sampel pada sebuah klaster lebih berdekatan.

Metode elbow bertujuan untuk menentukan elbow, yaitu jumlah K yang optimal. Untuk menentukan elbow, kita perlu melakukannya secara manual, yaitu dengan melihat titik dimana penurunan inersia tidak lagi signifikan. 

Pada contoh di bawah kita memiliki data yang dapat dibagi menjadi 4 klaster. 

![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/202101201401021fd247afd60d5456a02b536e99d3ebe7.png)

Kita akan mengaplikasikan metode elbow pada data di atas. Elbow berada di nilai K sama dengan 4, karena penurunan inersia pada K seterusnya tidak lagi signifikan (perubahannya nilainya kecil). Sehingga jumlah klaster yang optimal adalah 4.

![](https://lh5.googleusercontent.com/tZ1X0vxm68a6CeMzStvAaekfpV1Vz91uD3OfeYcN6nvhNLztG3kVkvwtX1xPoy_6EmDOji-pHib52AkFm-UyMR6buWIs618TrKBIUy5jR6e91WAs2I4Z_eCHemdiKiH4WfImosF3)