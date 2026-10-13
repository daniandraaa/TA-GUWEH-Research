# Analisis hasil eksperimen revisi TA

> **Pembaruan 30 September 2026 — prioritas V2:** sumber dataset telah diperiksa; sampling publik adalah **2.000 Hz**. Pengguna menyetujui koreksi tiga kelompok menjadi **LVOT / RVOT cusp / RVOT septal-free-wall-other**, dengan anggota pasien yang sama. Label historis “Non-OT” (LC/AC/RC) tidak sesuai anatomi sumber. Angka lama dipertahankan sebagai hasil historis. Mulai pekerjaan baru dari notebook **06**, mengikuti [panduan V2](PANDUAN_REVISI_V2.md). Training V2 belum dijalankan; GAN baru masih tahap berikutnya.


Audit artefak: 30 September 2026. Angka di bawah berasal dari run `20260929_grid9_v1`, bukan hasil eksperimen historis sebelum revisi. [Notebook visualisasi](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>) menghasilkan gambar dari berkas yang sama.

Jika membandingkan dengan angka 0,9 dari notebook lama, baca [audit skor lama versus revisi](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/AUDIT_SKOR_09_DAN_ARAH_PUBLIKASI.md>): angka tersebut adalah akurasi training, sedangkan angka utama revisi adalah macro-F1 pada pasien validation/test.

## Kelengkapan dan ketertelusuran

- Data utama: **230 pasien development** dan **59 pasien test historis**, tanpa ID pasien yang beririsan. Komposisi development LVOT 57, Non-OT 103, RVOT 70; test LVOT 14, Non-OT 21, RVOT 24.
- Lima fold pasien lengkap. Setiap pasien development muncul sebagai validation tepat sekali. ID yang dipakai fit scaler dan GAN pada tiap fold cocok dengan ID training fold; tidak ada pasien validation/test dalam sumber fit tersebut.
- GAN lima fold dan GAN final selesai 50 epoch. GAN final memakai 2.271 segmen asli dan menambah 675 segmen sintetis untuk menyamakan ketiga kelas menjadi 982 segmen per kelas.
- Fine-tuning selesai **135/135 training CV** (3 model × 9 skenario × 5 fold); baseline ulang **15/15 fold**. Tiga model final fine-tuning dan tiga baseline, semuanya seed 42, memiliki checkpoint serta hasil test.
- Setiap ringkasan CV cocok dengan rerata dan simpangan baku sampel lima file fold. Prediksi validation cocok dengan daftar pasien validation. Prediksi test berisi tepat 59 pasien unik dan menghasilkan kembali confusion matrix, macro-F1, serta AUC yang dilaporkan.
- SHAP tersedia untuk model utama fine-tuning dan model utama baseline; keduanya memakai 30 segmen development sebagai background dan menjelaskan 30 segmen test historis.

Sumber utama: [status artefak](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/status.json>), [manifest fold](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/folds/manifest.json>), [pilihan model utama](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/selected_model.json>), dan [tabel perbandingan](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/comparison/final_test_results.csv>). Field `pipeline.state=stopped` pada status mencatat proses otomatis lama yang dihentikan; jumlah hasil di atas berasal dari artefak training manual yang selesai kemudian.

## Hyperparameter terpilih dan hasil akhir

Seluruh konfigurasi dipilih berdasarkan **mean macro-F1 patient-level lima fold development**. SD dalam kolom CV adalah SD antar-fold, bukan SD dari lima training final. Satu final model per arsitektur dilatih dari awal pada seluruh development dengan seed 42 dan median best epoch skenario terpilih.

| Model | Skenario | Learning rate | Batch size | CV macro-F1 mean ± SD | Epoch final | Test macro-F1 fine-tuning | Test macro-F1 baseline | Selisih fine − baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN-Transformer Encoder | S09 | `1e-3` | 64 | 0,6806 ± 0,0439 | 24 | 0,6367 | 0,5282 | +0,1085 |
| CNN-PatchTST | S02 | `1e-4` | 32 | 0,6586 ± 0,0491 | 6 | 0,6775 | 0,3276 | +0,3499 |
| CNN-Hierarchical Transformer | S06 | `5e-4` | 64 | 0,6758 ± 0,0439 | 8 | 0,5270 | 0,6220 | −0,0950 |

**Model utama yang dibekukan dari CV adalah Encoder** (mean macro-F1 0,6806). Hierarchical dekat pada CV (0,6758), sehingga urutan keduanya perlu dibahas bersama variasi antar-fold. Pada test historis, **PatchTST mencatat macro-F1 tertinggi (0,6775)** dan benar pada 39/59 pasien; ini hasil pelaporan, bukan alasan mengganti model utama setelah melihat test. Encoder benar pada 37/59, Hierarchical 31/59. AUC test fine-tuning berturut-turut 0,7820; 0,8055; dan 0,7025.

Hasil fine-tuning tidak meningkat merata. Encoder dan PatchTST lebih baik daripada baseline pada test, sedangkan Hierarchical turun 0,0950 macro-F1. Baseline PatchTST tidak mengenali pasien Non-OT sama sekali (0/21); fine-tuning PatchTST mengenali 16/21. Pada model fine-tuning Encoder dan PatchTST, recall RVOT masih 10/24 (41,7%). Tabel nilai rata-rata saja akan menyembunyikan kesalahan kelas ini; gunakan confusion matrix dan grafik recall dari notebook 05.

Perbandingan baseline dan fine-tuning mencakup GAN serta perbedaan aturan training, sehingga selisihnya **tidak dapat diatribusikan hanya kepada learning rate/batch size**. Kurva loss GAN menunjukkan proses optimasi, bukan bukti otomatis bahwa sinyal sintetis memiliki mutu klinis memadai.

**Diagnostik GAN yang perlu ditindaklanjuti:** pada Lead II fold 1, median rata-rata selisih absolut antar-sampel GAN dibanding real adalah 0,29635 vs 0,00853 untuk LVOT (**34,8×**) dan 0,22878 vs 0,00657 untuk RVOT (**34,8×**). Audit diperluas ke lima fold, GAN final, serta 12 sadapan: rasio kehalusan sintetis/real berada pada **10,3–54,7×**. Pada GAN final, Lead II berada pada 23,3–24,9×. Lihat [heatmap seluruh fold](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06c_diagnostik_GAN_semua_fold.png>) dan [CSV terukur](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06c_diagnostik_GAN_semua_fold.csv>).

Pemeriksaan tambahan hubungan sadapan ekstremitas (misalnya `III = II − I`) menunjukkan median galat absolut GAN final **29,4–87,2×** lebih besar daripada real DWT, tergantung hubungan dan kelas; pola ini juga muncul pada lima fold. Lihat [heatmap konsistensi sadapan](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06d_konsistensi_sadapan.png>) dan [CSV](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06d_konsistensi_sadapan.csv>). Pemeriksaan ini menguatkan bahwa mutu keluaran GAN perlu diteliti; belum membuktikan bahwa GAN menyebabkan perubahan F1 atau bahwa semua aspek morfologi klinisnya salah. Periksa konstruksi 12 sadapan dan preprocessing bersama ahli ECG sebelum menyatakan sinyal sintetis realistis. Jangan mengubah run yang sudah dibekukan; eksperimen koreksi harus menggunakan run ID dan protokol baru.

## Yang perlu dikerjakan setelah lima notebook training

1. Jalankan [notebook 05](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>) sel demi sel. Sebanyak 19 grafik PNG sudah dihasilkan dan tersedia di folder `results/finetune_revision/20260929_grid9_v1/visualisasi/`. Ambil gambar yang diperlukan untuk Bab IV. Pisahkan tabel CV dan hasil test; jelaskan model utama berdasarkan CV dan hasil test tertinggi sebagai temuan yang berbeda.
2. Terapkan [draf revisi Bab III](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/REVISI_METODOLOGI_BAB_III.md>) ke naskah TA yang dapat diedit: grid 9 kombinasi, 5-fold berbasis pasien, scaler/GAN hanya pada training fold, pemilihan dari CV, retraining satu model final per arsitektur, dan inferensi 20% tanpa CV. Proposal yang tersedia di project masih PDF asli.
3. Tinjau diagnostik GAN seluruh fold dan 12 sadapan bersama pembimbing/ahli ECG. Telusuri penyebab perbedaan kehalusan dan hubungan sadapan sebelum memasukkan klaim mutu sinyal sintetis ke naskah. Hasil klasifikasi saat ini tetap dilaporkan secara transparan.
4. Tulis Bab IV/V dari hasil aktual: parameter terpilih, mean ± SD antar-fold, perbandingan baseline, confusion matrix/recall per kelas, hasil negatif Hierarchical, diagnostik GAN, dan keterbatasan. Verifikasi sumber frekuensi sampling sebelum menyebut 5.000 sampel sebagai durasi tertentu.
5. Nyatakan secara eksplisit bahwa 59 pasien test adalah **hold-out historis yang sudah pernah dipakai pada eksperimen lama**. Protokol revisi ini memilih parameter tanpa memakai test, tetapi tidak mengubah riwayat pemakaian data tersebut. Evaluasi pada pasien eksternal baru diperlukan bila ingin mengklaim generalisasi yang benar-benar belum diuji.
6. Jika pembimbing meminta stabilitas lima pengulangan, tetapkan seed dan rancangan analisis sebelum retraining independen. Jangan mengulang inferensi pada bobot yang sama atau memilih seed dengan skor test tertinggi. Saat ini hasil final memakai satu seed (42). Analisis Summit/unknown juga belum dijalankan pada alur manual dan merupakan tambahan tanpa metrik akurasi tiga kelas.

SHAP fine-tuning utama menempatkan V4, V5, V6 sebagai tiga sadapan dengan mean absolute SHAP tertinggi pada 30 segmen yang dijelaskan. Ini deskripsi model dan sampel tersebut, bukan penanda klinis tervalidasi atau penjelasan kausal.
