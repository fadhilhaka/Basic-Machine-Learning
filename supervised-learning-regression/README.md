## Regression

Regresi adalah salah satu teknik ML yang mirip dengan klasifikasi. Bedanya pada klasifikasi, sebuah model ML memprediksi sebuah kelas, sedangkan model regresi memprediksi bilangan kontinu. Bilangan kontinu adalah bilangan numerik.

Model regresi memprediksi sebuah nilai berdasarkan atribut yang tersedia.

| Lama Bekerja | Industri | Tingkat Pendidikan | Gaji |
|--------------|----------|--------------------|------|
| 6 tahun | Marketing | SMA	| 8.000.000 | 
| 12 tahun | IT | S1 | 16.000.000 |
| 8 tahun |	Kesehatan |	S2 | 20.000.000 | 
| 5 tahun |	IT | SMK | ? | 
| 6 tahun |	Marketing |	S2 | 14.000.000 | 
| 21 tahun | Perbankan | S3 | 35.000.000 | 
| 3 tahun |	IT | S1 | 10.000.000 |

Pada contoh data di atas, model regresi akan memprediksi gaji berdasarkan atribut lama bekerja, industri, dan tingkat pendidikan. Gaji adalah contoh dari bilangan kontinu, di mana gaji tak memiliki kategori-kategori yang terbatas.

Pada submodul ini jenis regresi yang akan dibahas adalah regresi linier. Selain regresi linier terdapat juga jenis regresi lain seperti regresi polinomial, lasso regression, stepwise regression dan sebagainya.

[Jenis-jenis regresi](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/)

### Linear Regression

Contoh paling terkenal dari regresi linier adalah memperkirakan harga rumah berdasarkan fitur yang terdapat pada rumah seperti luas rumah, jumlah kamar tidur, lokasi dan sebagainya.

Regresi linier cocok dipakai ketika terdapat hubungan linear pada data. Namun untuk implementasi pada kebanyakan kasus, ia kurang direkomendasikan. Sebabnya, regresi linier selalu mengasumsikan ada hubungan linier pada data.

1. Secara sederhana regresi linear adalah teknik untuk memprediksi sebuah nilai dari variable Y (variabel dependen) berdasarkan beberapa variabel tertentu X (variabel independen) jika terdapat hubungan linier antara X dan Y.
2. Hubungan antara hubungan linier dapat direpresentasikan dengan sebuah garis lurus (disebut garis regresi). Ilustrasi hubungan linier dapat dilihat pada gambar di mana data-data cenderung memiliki pola garis lurus.
   ![](https://lh4.googleusercontent.com/Jfj-v6lFHv9gBRj_G2dnKXDBgk1by09dKP3FcptRcjOdpPHdj9oZ7l4RMyDovi5jgF8icA4XDkRMjdLFA1XI3rbZ72nqnaJ0z2PeW_V4PQFhVaXSk_p6pKg1MhTwLaGnV_bOpQbK)
3. Ketika sebuah garis regresi digambar, beberapa data akan berada pada garis regresi dan beberapa yang lainnya akan berada di dekat garis tersebut. Sebabnya, garis regresi adalah sebuah model probabilistik dan prediksi kita adalah perkiraan. Jadi tentu akan ada eror/penyimpangan terhadap nilai asli dari variabel Y. Pada gambar di bawah, garis merah yang menghubungkan data-data ke garis regresi merupakan eror. Semakin banyak eror artinya model regresi itu belum optimal.
   ![](https://lh6.googleusercontent.com/cXIr8HD3j5RYSep5gmdszcn4lWwWTu3iFjPhox7r3M_SB3aWTrYUFhZW0gIZsubaPzEa9e3CYRuYNZtSD_9cIucMPF3YFi_rh_3TfslNYNBgGNMhAIdt4RXoFnD8R8tuhA1Dmd3t)

[Latihan SKLearn Linear Regression](https://jp-tok.dataplatform.cloud.ibm.com/analytics/notebooks/v2/f249deb9-6028-4c9f-9524-77439ee599c0?projectid=f052c4d4-84ba-485b-8bd0-827f0b83f55f&context=cpdaas)

### Logistic Regression

Logistic regression dikenal juga sebagai logit regression, maximum-entropy classification, dan log-linear classification merupakan salah satu metode yang umum digunakan untuk klasifikasi. Pada kasus klasifikasi, logistic regression bekerja dengan menghitung probabilitas kelas dari sebuah sampel. 

Sesuai namanya, logistic regression menggunakan fungsi logistik seperti di bawah untuk menghitung probabilitas kelas dari sebuah sampel. Contohnya sebuah email memiliki probabilitas 78% merupakan spam maka email tersebut termasuk dalam kelas spam. Dan jika sebuah email memiliki <50% probabilitas merupakan spam, maka email tersebut diklasifikasikan bukan spam.

![](https://lh5.googleusercontent.com/YhQzESOduIjO5aPj-HW8FbutIRqMhAXTstdPmJWkeY81_Nh2O22YKvVsdAgmLY-i4hl553N_05GmdP1kpdD9Sf9TEU8xbO-m0Ef1AdxLKKt0sIkTk7zvIduiWgBejQ-tw8x5p2aF)

[Social Network Ads](https://www.kaggle.com/dragonheir/logistic-regression)

[Latihan SKLearn Logistic Regression](https://jp-tok.dataplatform.cloud.ibm.com/analytics/notebooks/v2/f249deb9-6028-4c9f-9524-77439ee599c0?projectid=f052c4d4-84ba-485b-8bd0-827f0b83f55f&context=cpdaas)

>Model regresi linier adalah salah satu model machine learning yang paling sederhana. Model ini memiliki kompleksitas rendah dan bekerja sangat baik pada dataset yang memiliki hubungan linier. 

[Get the full path of the current file's directory](https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory)