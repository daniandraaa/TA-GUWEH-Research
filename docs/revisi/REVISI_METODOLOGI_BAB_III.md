# Draf revisi metodologi Bab III

> **Pembaruan 30 September 2026 — prioritas V2:** sumber dataset telah diperiksa; sampling publik adalah **2.000 Hz**. Pengguna menyetujui koreksi tiga kelompok menjadi **LVOT / RVOT cusp / RVOT septal-free-wall-other**, dengan anggota pasien yang sama. Label historis “Non-OT” (LC/AC/RC) tidak sesuai anatomi sumber. Angka lama dipertahankan sebagai hasil historis. Mulai pekerjaan baru dari notebook **06**, mengikuti [panduan V2](PANDUAN_REVISI_V2.md). Training V2 belum dijalankan; GAN baru masih tahap berikutnya.


Draf ini menyesuaikan implementasi revisi dengan arahan bimbingan. Proposal asli tetap menjadi sumber rujukan; teks berikut siap digunakan untuk memperbarui bagian skenario fine-tuning, pembagian data, dan Tabel 3.3.

**Catatan terminologi sebelum dipindahkan ke naskah:** implementasi classifier dalam run ini dimulai dari bobot acak, bukan dari checkpoint pretrained. Istilah “fine-tuning” di bawah mengikuti penamaan skenario proposal; secara teknis prosesnya adalah *training dari awal dengan tuning hiperparameter dan augmentasi GAN*. Selaraskan judul, tujuan, dan istilah bersama pembimbing, atau tambahkan tahap transfer learning yang benar jika fine-tuning bobot pretrained memang dimaksudkan.

## Pembagian pasien dan validasi

Dataset eksperimen utama mencakup 289 pasien dengan tiga kelas target, yaitu LVOT, Non-OT, dan RVOT. Pembagian historis menghasilkan 230 pasien sebagai data pengembangan dan 59 pasien sebagai hold-out pengujian. Seluruh segmen yang berasal dari seorang pasien ditempatkan pada bagian yang sama. Kelompok Summit dan pasien tanpa label sublokasi tidak dimasukkan dalam evaluasi terawasi tiga kelas utama.

Lima fold dibentuk pada pasien data pengembangan menggunakan StratifiedGroupKFold dengan seed 42. Pada setiap iterasi, empat fold digunakan untuk training dan satu fold digunakan untuk validation. Manifest pasien yang sama digunakan pada seluruh konfigurasi serta ketiga arsitektur. Setiap pasien pengembangan menjadi validation tepat sekali. Test tidak menjadi bagian dari fold.

Sinyal hasil DWT disegmentasi menjadi 5.000 sampel dengan overlap 50%. Sampling rate 1.000 Hz mengikuti asumsi kode EDA historis dan perlu dicocokkan dengan dokumentasi sumber dataset. Dengan asumsi itu, panjang segmen setara lima detik. StandardScaler di-fit hanya dari segmen training fold; parameter tersebut digunakan untuk mentransformasi validation fold. Pada training final, scaler di-fit ulang dari seluruh data pengembangan.

## Augmentasi GAN

Pada skenario fine-tuning, conditional GAN dilatih hanya menggunakan data asli dari training fold yang sedang digunakan. Data sintetis ditambahkan untuk menyeimbangkan jumlah segmen tiap kelas training terhadap kelas terbesar pada fold tersebut. Validation fold dan test menggunakan data asli. Generator serta data sintetis satu fold dapat dipakai kembali pada seluruh skenario dan model dalam fold yang sama karena konfigurasi GAN dibuat tetap.

GAN menggunakan 50 epoch, batch size 64, latent dimension 100, embedding 50, optimizer Adam dengan learning rate 0,0002 dan beta 0,5/0,999. Sinyal training yang sudah dinormalisasi dibatasi pada rentang [-1, 1] setelah pembagian skala 5, sesuai keluaran Tanh generator. Keluaran sintetis kemudian dikembalikan ke skala Z-score. Pada training final, GAN dilatih ulang menggunakan seluruh data pengembangan. Data validation ataupun test tidak digunakan untuk melatih GAN.

Audit setelah training menemukan [perbedaan kehalusan dan hubungan sadapan yang besar](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/PRIORITAS_PERBAIKAN_DAN_EKSPERIMEN.md>) antara sinyal GAN dan real. Jelaskan penggunaan GAN sebagai prosedur yang diuji, tanpa menyatakan bahwa kualitas morfologinya telah tervalidasi. Jika GAN diperbaiki, hasilnya merupakan eksperimen baru dengan protokol dan run ID baru.

## Pencarian hiperparameter

Grid search dibatasi pada tiga learning rate dan tiga batch size. Setiap pasangan diuji melalui lima fold pada CNN-Transformer Encoder, CNN-PatchTST, dan CNN-Hierarchical Transformer. Setiap model menghasilkan 45 pelatihan CV, sehingga ketiga model menghasilkan 135 pelatihan CV. Angka ini tidak mencakup GAN, baseline, atau retraining final.

| Parameter | Konfigurasi revisi |
|---|---|
| Learning rate awal | 0,0001; 0,0005; 0,001 |
| Batch size model klasifikasi | 16; 32; 64 |
| Optimizer fine-tuning | Adam |
| Weight decay | 0,0001 |
| Maksimum epoch CV | 100 |
| Early stopping patience | 10 |
| Scheduler | Cosine Annealing; T_max 100; minimum LR 0,000001 |
| Gradient clipping | 1,0 |
| Loss fine-tuning | Cross-entropy tanpa class weighting dan tanpa label smoothing |
| Arsitektur | Definisi identik dengan baseline; tidak menambahkan pooling atau fusion baru |
| Dropout Transformer / classifier | 0,2 / 0,3, dibuat tetap |
| Validation | Data asli per pasien; tidak menggunakan data sintetis |

Batch size diperbarui dari 64/128/256 dalam proposal awal menjadi 16/32/64. Penyesuaian mempertimbangkan ukuran input 12 sadapan × 5.000 sampel, kebutuhan memori, dan rentang eksperimen pendahuluan. Hasil pendahuluan tidak digunakan untuk menetapkan parameter final; konfigurasi final ditentukan dari grid lengkap dengan prosedur validasi yang diperbaiki.

Baseline tetap menggunakan konfigurasi awal kode historis: AdamW, learning rate 0,001, batch size 32, maksimum 50 epoch, patience 10, dan cosine scheduler T_max 50. Class weights dihitung hanya dari training fold. Baseline tidak menggunakan GAN atau grid search. Arsitektur, manifest fold, normalisasi per fold, dan metode evaluasi patient-level dibuat sama dengan skenario fine-tuning. Perbedaan hasil membandingkan keseluruhan skenario baseline dengan GAN dan konfigurasi fine-tuning; perbandingan tersebut tidak mengisolasi pengaruh learning rate saja.

## Pemilihan konfigurasi dan training final

Probabilitas seluruh segmen seorang pasien dirata-ratakan, kemudian kelas dengan probabilitas tertinggi menjadi prediksi pasien. Metrik utama untuk pemilihan konfigurasi adalah mean patient-level macro-F1 dari lima validation fold. Accuracy, precision, sensitivity macro, specificity macro, dan AUC-ROC macro one-vs-rest juga dicatat. Standar deviasi antar-fold dihitung sebagai sample standard deviation dengan ddof=1. Jika mean macro-F1 sama, konfigurasi dengan standar deviasi lebih kecil diutamakan, kemudian digunakan urutan ID skenario.

Konfigurasi terbaik dipilih terpisah untuk setiap arsitektur. Arsitektur utama ditetapkan dari skor CV sebelum evaluasi test. Model final dilatih dari inisialisasi baru pada seluruh 230 pasien pengembangan menggunakan konfigurasi terpilih. Durasi final menggunakan median best epoch pada lima fold konfigurasi terpilih, tanpa batas minimum epoch tambahan. Kebijakan scheduler CV dipertahankan saat final training. Seed utama final adalah 42.

Model fold berfungsi sebagai bukti validasi. Keluaran final berupa satu checkpoint hasil retraining per arsitektur, berikut scaler, label mapping, konfigurasi, seed, serta riwayat eksperimen. Lima model fold tidak dipakai sebagai ensemble final.

## Evaluasi dan interpretasi

Inferensi akhir dilakukan pada hold-out 59 pasien tanpa k-fold dan tanpa pembaruan bobot. Model, konfigurasi, epoch, dan seed ditetapkan sebelum test. Confusion matrix, classification report, dan metrik patient-level dilaporkan untuk ketiga model. Pengulangan lima seed bersifat opsional dan harus ditetapkan sebelum evaluasi; implementasi utama menggunakan satu seed final.

Hold-out tersebut sudah pernah dievaluasi dalam eksperimen terdahulu, termasuk pemilihan model dari skor test. Hasil revisi harus menyebutnya sebagai hold-out historis, bukan mengklaim sebagai dataset yang belum pernah diamati. Pembekuan protokol revisi menghindari pemilihan baru berdasarkan test, tetapi tidak menghapus riwayat penggunaan data.

SHAP GradientExplainer diterapkan pada model utama yang dipilih dari CV. Background berasal dari data pengembangan. Kontribusi sadapan dan waktu dihitung terhadap masukan sinyal DWT yang sudah dinormalisasi, bukan terhadap koefisien dekomposisi DWT secara langsung. Prediksi kelompok Summit dan unknown merupakan analisis eksploratif tanpa perhitungan akurasi tiga kelas utama.
