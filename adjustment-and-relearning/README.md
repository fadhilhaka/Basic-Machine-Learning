# Adjustment and Re-learning

## Adjustment

Umumnya sebuah model yang di-deploy kinerjanya akan turun seiring waktu. Karena model akan terus menemui lebih banyak data baru seiring waktu. 

Hal tersebut akan menyebabkan akurasi model menurun. Misalnya sebuah model untuk memprediksi harga rumah yang dikembangkan dengan data pada tahun 2010. Model yang dilatih pada data pada tahun tersebut akan menghasilkan prediksi yang buruk pada data tahun 2020.

Untuk mengatasi masalah ini, ada 2 teknik dasar untuk menjaga agar model selalu bisa belajar dengan data-data baru. Dua teknik tersebut yaitu manual retraining dan continuous learning.

### Manual Retraining

Teknik pertama adalah melakukan ulang proses pelatihan model dari awal. Di mana data-data baru yang ditemui di tahap produksi akan digabung dengan data lama. Lebih lanjut, model dilatih ulang dari awal sekali menggunakan data lama dan data baru.

Bayangkan ketika kita harus melatih ulang model dalam jangka waktu harian atau bahkan mingguan. Sesuai yang Anda bayangkan, proses ini akan sangat memakan waktu.

Namun, manual retraining juga memungkinan kita menemukan model-model baru atau atribut-atribut baru yang menghasilkan performa lebih baik.

### Continuous Learning

Teknik kedua untuk menjaga model kita up-to-date adalah continuous learning yang menggunakan sistem terotomasi dalam pelatihan ulang model. Alur dari continuous learning yaitu:

1. Menyimpan data-data baru yang ditemui pada tahap produksi. Contohnya ketika sistem mendapatkan harga emas naik, data harga tersebut akan disimpan di database.
2. Ketika data-data baru yang dikumpulkan cukup, lakukan pengujian akurasi dari model terhadap data baru.
3. Jika akurasi model menurun seiring waktu, gunakan data baru, atau kombinasi data lama dan data baru untuk melatih dan men-deploy ulang model.

Sesuai namanya, 3 proses di atas dapat terotomasi sehingga kita tidak perlu melakukannya secara manual.