# Latihan Membuat Model di Watson Studio

* Buka halaman project.
* Pilih tab settings.
* Scroll ke bawah sampai ke bagan Associated service.
* Klik Add service, dan pilih Watson.
* Halaman Associate service akan terbuka. Jika telah ada service sebelumnya dengan type Machine Learning, Anda tinggal berikan tanda centang. Namun jika belum, Anda dapat klik tombol New service kemudian pilih service Machine learning dan lengkapi informasi yang dibutuhkan.
* Beri tanda centang pada service yang telah ada lalu klik tombol Associate service.
* Kita akan menggunakan dataset iklan di sosial media untuk memprediksi apakah seseorang akan membeli setelah melihat iklan sebuah produk. Untuk dataset yang akan kita pakai dapat Anda unduh pada [tautan berikut](https://www.kaggle.com/dragonheir/logistic-regression). 
* Unggah dataset Anda dengan drag and drop atau pilih berkas langsung dari penyimpanan lokal Anda.
* Untuk melihat data yang baru kita unggah, kita cukup klik nama dataset yang ada pada bagian Data Assets.
* Untuk menampilkan statistik interaktif dari data kita, kita bisa menuju ke tab profile. Pada tab profile pilih ‘Create Profile’. Proses pembuatan profile akan memakan beberapa waktu.
* Jika proses selesai, grafik interaktif serta statistik dari setiap kolom dapat kita lihat. Grafik batang menampilkan distribusi setiap objek dari setiap kolom. Rata-rata serta deviasi standar setiap kolom juga ditampilkan. Pada kolom Purchased yang merupakan label dapat dilihat bahwa distribusi kelasnya tidak seimbang. Jumlah kelas ‘0’ (tidak membeli) sekitar 2 kali lipat dari kelas ‘1’ (membeli). Kita akan melihat bagaimana performa model dari Watson Machine Learning dengan data yang tidak seimbang ini.
  ![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/202009281328443b74ae313df35fe3adcf9826e485cea4.jpeg)
* Selanjutnya mengolah dataset, kembali ke tab preview lalu klik tombol Refine.
* Pada laman refine Anda dapat melakukan operasi-operasi pengolahan data seperti membersihkan data, mengubah tipe data, dan beberapa operasi lain. Untuk melihat operasi apa saja yang dapat dilakukan dengan refine, klik tombol Operation yang berwarna biru.
* Mulai mengolah data dengan menghilangkan kolom yang tidak relevan. Kolom User ID tidak memiliki pengaruh terhadap prediksi model sehingga kita perlu menghilangkan dataset tersebut. Caranya adalah dengan klik tombol 3 titik yang ada di kanan atas kolom User ID dan pilih ‘Remove’.
* Ketika kolom User ID telah dihilangkan, akan muncul keterangannya di bagian Steps yang berada di sebelah kanan. Steps menunjukkan operasi apa saja yang telah kita lakukan pada dataset, memudahkan kita untuk melacak operasi-operasi yang telah kita lakukan. Dapat kita lihat bahwa ada 2 operasi pada bagian Steps yaitu operasi penghapusan kolom User ID dan operasi konversi tipe kolom. Operasi konversi tipe kolom ini mengubah tipe data dari setiap kolom agar lebih sesuai. Misalnya ada sebuah kolom ‘Gaji’ yang berisi gaji yang nilai-nilainya berupa angka, namun dalam format string. Kolom tersebut akan diubah menjadi tipe Integer secara otomatis.
* Simpan dataset ini dengan klik tombol jobs yang terdapat di dalam kotak berwarna merah, lalu pilih ‘Save and create a job’.
* Pada halaman Create a job isi kolom pada bagian define details lalu klik Next. Isilah nama job dan deskripsi sesuai keinginan Anda.
* Pada bagian Configure Anda dapat langsung klik Next.
* Pada bagian selanjutnya biarkan Schedule off lalu klik Next.
* Di bagian Review and create pilih ‘Create and run’ lalu tunggu sampai status Runs menjadi Completed.
* Kembali ke halaman proyek Anda dan buka tab Assets. Jika tahapan sebelumnya berhasil maka akan muncul data baru pada bagian Data Assets. Data yang baru adalah data yang telah proses sebelumnya. Jika Anda klik data terbaru maka dapat Anda lihat bahwa data tersebut tidak memiliki kolom User ID.
* Untuk membuat model ML pada Watson Studio kita dapat menggunakan fitur AutoAI. Fitur AutoAI secara otomatis memilih, menyeleksi, serta mencari sendiri parameter model yang paling optimal.
* Untuk menggunakan fitur AutoAI, buka tab Assets pada halaman proyek kita dan klik tombol Add to Project yang berwarna biru.
* Di bagian New AutoAI experiment isi kolom yang ada. Untuk kolom Watson Machine Learning Service Instance pilih WatsonMachineLearning. Lalu klik Create.
* Kemudian Anda akan diminta untuk menambahkan dataset yang akan diproses oleh model Anda. Pilih Select from project.
* Lalu pilih dataset yang telah diproses yaitu dataset yang tidak memiliki kolom UserID. Klik select asset.
* Pada bagian Configure details pilih kolom yang merupakan target, atau kolom yang berisi label pada dataset kita yaitu kolom Purchased. AutoAI cukup cerdas untuk mengetahui bahwa dataset kita merupakan dataset untuk klasifikasi biner. Klik tombol Run Experiment.
* Proses AutoAI akan berjalan selama beberapa waktu. Tunggu hingga selesai.
* Jika proses pelatihan model selesai, tampilannya akan menjadi seperti di bawah.
  ![](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20200928132848d6079f848e71e8cd3fcf42f1d515718b.jpeg)
* Jika Anda scroll ke bawah, Anda dapat melihat bahwa AutoAI telah melakukan 7 kali pelatihan 2 buah model berbeda yaitu XGBClassifier dan Extra Tree Classifier. Watson Studio dengan cerdas memilih model-model terbaik yang sesuai dengan dataset yang kita masukkan. Hasilnya pun sangat memuaskan dengan akurasi di kisaran 0.883 sampai 0.915.

Masih ingat dengan latihan kita pada modul Latihan SKLearn Logistic Regression? Di modul tersebut kita menulis kode untuk membuat model ML dengan dataset yang sama dengan latihan kali ini. Pada latihan tersebut akurasi yang kita dapat sekitar 0.63. Sedangkan dengan menggunakan IBM Watson Studio, tanpa menulis kode satu baris pun kita mendapatkan model dengan akurasi yang jauh lebih tinggi.

Namun perlu diperhatikan bahwa model yang dihasilkan oleh IBM Watson Studio tidak selalu mengungguli model yang ditulis manual. IBM Watson Studio lebih tepat digunakan untuk menguji apakah sebuah proyek ML layak dikembangkan atau tidak. Salah satu implementasinya adalah menguji apakah sebuah dataset memiliki kualitas yang baik dan dapat dipakai. Seperti pada latihan kali ini, dataset yang kita pakai dapat menghasilkan model dengan akurasi 0.89 ke atas. Maka kita mendapat gambaran bahwa dataset ini layak dipakai dan model ML layak dikembangkan. Selanjutnya kita dapat mulai mengembangkan model ML dengan menulis kode manual yang memiliki fleksibilitas dan kompleksitas lebih baik, khususnya jika proyek ML yang kita kembangkan adalah proyek besar.