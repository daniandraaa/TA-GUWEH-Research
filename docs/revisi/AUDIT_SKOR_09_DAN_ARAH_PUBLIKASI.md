# Audit skor 0,9 dan arah eksperimen publikasi

> **Pembaruan 30 September 2026 — prioritas V2:** sumber dataset telah diperiksa; sampling publik adalah **2.000 Hz**. Pengguna menyetujui koreksi tiga kelompok menjadi **LVOT / RVOT cusp / RVOT septal-free-wall-other**, dengan anggota pasien yang sama. Label historis “Non-OT” (LC/AC/RC) tidak sesuai anatomi sumber. Angka lama dipertahankan sebagai hasil historis. Mulai pekerjaan baru dari notebook **06**, mengikuti [panduan V2](PANDUAN_REVISI_V2.md). Training V2 belum dijalankan; GAN baru masih tahap berikutnya.


Audit: 30 September 2026. Dokumen ini membandingkan artefak lama yang masih tersimpan dengan run revisi `20260929_grid9_v1`. Tidak ada training baru dalam audit ini.

## Angka 0,9 berasal dari mana?

[Grafik training lama](</Users/daniandraaa/Desktop/TA GUWEH/finetune_results/final_training_curves.png>) dan sel 15 [notebook fine-tuning historis](</Users/daniandraaa/Desktop/TA GUWEH/arsip/notebooks_lama/03_Finetune_CNN_Transformer.ipynb>) menampilkan `TrPatAcc`: akurasi pada **230 pasien development yang juga dipakai untuk melatih model final**. Nilainya pada epoch akhir adalah Encoder **0,948**, PatchTST **0,957**, dan Hierarchical **0,917**. Ini bukan F1 maupun akurasi pada 59 pasien test.

Kode lama menghitung `TrPatAcc` dengan `final_train_eval_loader` dari `X_train_scaled, y_train_all`, lalu menghitung test di loader terpisah. Karena itu angka training dan test tidak boleh disandingkan seolah-olah satu jenis evaluasi. Pada run revisi, `history.csv` model final mencatat *segment-level training accuracy* akhir Encoder 0,9206, PatchTST 0,8975, Hierarchical 0,9236; unit segment ini juga tidak dapat dibandingkan langsung dengan `TrPatAcc` lama yang patient-level. Kedua run dapat mencapai akurasi training sekitar 0,9 tanpa mencapai 0,9 pada pasien test.

## Perbandingan adil pada 59 pasien test

Kedua tabel CSV mencatat **patient-level macro-F1** pada split test historis yang sama. Tabel ini membandingkan metrik dan unit yang sama, walau resep training dan arsitektur berbeda.

| Model | F1 test lama | F1 test revisi | Selisih | Accuracy test lama | Accuracy test revisi |
|---|---:|---:|---:|---:|---:|
| Encoder | 0,5981 | 0,6367 | +0,0386 | 0,5763 | 0,6271 |
| PatchTST | 0,6200 | 0,6775 | +0,0575 | 0,6102 | 0,6610 |
| Hierarchical | 0,6102 | 0,5270 | −0,0831 | 0,5932 | 0,5254 |

Sumber: [hasil lama](</Users/daniandraaa/Desktop/TA GUWEH/finetune_results/final_test_results.csv>) dan [hasil revisi](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/comparison/final_test_results.csv>). Pada eksperimen lama, gap PatchTST antara accuracy training patient-level 0,957 dan accuracy test patient-level 0,610 menunjukkan generalisasi jauh lebih lemah daripada kecocokan pada data training. Gap serupa muncul pada dua model lain. Ini indikasi, bukan diagnosis tunggal penyebabnya.

## Apa yang berubah?

1. **Protokol CV dan augmentasi:** kode lama fit scaler serta GAN pada seluruh 80% development sebelum CV. Data GAN yang sama lalu ditambahkan pada training setiap fold. Revisi fit scaler dan GAN hanya pada training fold terkait; validation tetap pasien asli. Perubahan ini menutup jalur informasi antar-fold, tetapi besarnya pengaruh pada skor belum diisolasi.
2. **Arsitektur:** fine-tuning historis memakai attention pooling pada Encoder/PatchTST dan attention pooling serta fusion tambahan pada Hierarchical. Revisi memakai definisi arsitektur baseline yang sama untuk kelompok baseline dan kelompok yang dituning, dengan mean pooling. Penurunan Hierarchical dapat terkait perubahan ini, tetapi perlu ablation untuk membuktikannya.
3. **Konfigurasi dan aturan training:** run lama memilih dari empat konfigurasi terarah yang sekaligus mengubah learning rate, batch size, dropout dan weight decay; Encoder/PatchTST memilih LR `2e-4`, yang tidak ada di grid revisi. Revisi menguji sembilan pasangan LR × batch dengan parameter lain tetap. Optimizer fine-tuning berubah dari AdamW/label smoothing 0,05 menjadi Adam/tanpa label smoothing; dropout dan aturan epoch final juga berubah. Jadi efek satu perubahan tidak dapat dibaca dari tabel hasil akhir.
4. **Pemilihan dan pelaporan:** kode lama menandai `best_final` dari F1 test. Revisi memilih konfigurasi dan model utama dari CV development sebelum test. Test historis sudah pernah dipakai dalam eksperimen lama, sehingga evaluasi revisi tetap harus disebut evaluasi pada *hold-out historis*.

## Apakah 0,6775 adalah batas maksimal?

Tidak ada bukti bahwa model atau dataset ini sudah mencapai batas kemampuan. Juga tidak ada dasar dari artefak project untuk mengklaim F1 test 0,9. Sampel test 59 pasien cukup kecil untuk membuat estimasi per kelas peka terhadap beberapa kesalahan; pada fine-tuning Encoder/PatchTST, RVOT benar 10/24 pasien. Skor yang lebih tinggi di training tidak menyelesaikan masalah generalisasi.

## Urutan penelitian tambahan yang dapat dipertanggungjawabkan

1. **Audit label dan sinyal:** pastikan label pasien, sadapan, frekuensi sampling, segmentasi, dan preprocessing benar. [Audit GAN seluruh fold dan 12 sadapan](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06c_diagnostik_GAN_semua_fold.png>) menemukan perubahan antar-sampel sintetis 10,3–54,7× sinyal real; [hubungan sadapan ekstremitas](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06d_konsistensi_sadapan.png>) juga jauh kurang konsisten. Tinjau morfologi bersama ahli ECG sebelum mengklaim augmentasi realistis.
2. **Ablation pada run ID baru:** bandingkan *real-only*, penanganan ketidakseimbangan tanpa GAN (misalnya class weighting), dan GAN dengan split pasien, arsitektur, metrik, serta anggaran tuning yang sebanding. Pilih rancangan dan seed di development; jangan menentukan pemenang dari test historis. Jika ingin menguji attention pooling/fusion lama, jadikan itu eksperimen arsitektur tersendiri, bukan efek learning rate.
3. **Ukur ketidakpastian dan kesalahan:** laporkan confusion matrix, sensitivity/recall tiap kelas, interval kepercayaan berbasis pasien, serta analisis kasus RVOT yang salah. Ulangi training dengan seed yang ditetapkan lebih dahulu bila ingin menilai stabilitas. Jangan memilih seed berdasarkan F1 test.
4. **Validasi pada pasien baru:** bila tersedia, siapkan cohort eksternal atau prospektif yang belum pernah dilihat selama desain eksperimen. Ini jauh lebih informatif untuk klaim generalisasi daripada mengoptimalkan 59 pasien test historis.
5. **Benahi istilah ilmiah:** kode final saat ini membuat model baru melalui `build_model(...)` dan melatihnya dari bobot awal, tanpa memuat bobot classifier pretrained. Secara teknis ini *pelatihan dari awal dengan tuning hiperparameter dan augmentasi GAN*. Jika naskah menyebut *fine-tuning* dalam arti transfer learning, perlu checkpoint pretrained dan prosedur adaptasi bobot yang nyata; jika tidak, jelaskan penggunaan istilah secara tepat kepada pembimbing/reviewer.

Untuk naskah AI diagnostik, gunakan pedoman pelaporan [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8); untuk pengembangan model prediksi, [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) relevan menurut lingkup studi. Target jurnal Q1 adalah kategori dan tahun pemeringkatan jurnal, bukan ambang F1=0,9. [Panduan Scopus tentang quartile](https://www.elsevier.support/scopus/answer/how-do-i-use-the-scopus-sources-feature) dapat dipakai saat memilih jurnal. Tidak ada jaminan penerimaan Q1 dari skor tunggal; validitas desain, kontribusi ilmiah, serta pelaporan transparan tetap harus dinilai.
