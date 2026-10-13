# Analisis hasil V11 — tiga arm fitur selesai, 7 Oktober 2026

Notebook **60 dan 61 selesai masing-masing 11/11 sel kode tanpa error tersimpan**. Tiga arm fitur menyelesaikan seluruh lima fold. Empat arm encoder/fusion menampilkan `TERTUNDA` karena cache representasi V10 belum tersedia; ini bukan hasil gagal classifier atau skor nol.

Run yang diperiksa: `20261007_morphology_regularized_features_v11_r1`, experiment hash `598a7ed340c33632980d23fe9436a259fbbde25a64cc9f5760950f548173aead`.

**Kesimpulan saat ini:** morfologi antarlead memberi informasi tambahan dibanding QRS/RR saja. Penambahan RR pada morfologi belum memberi peningkatan rata-rata lima fold. Masalah generalisasi dan pemisahan kedua kelompok RVOT masih nyata. Hasil ini tidak membuktikan batas maksimum dataset atau keberhasilan fusion CNN–Transformer.

## 1. Pekerjaan yang sudah selesai

| Notebook | Hasil aktual | Status |
|---|---|---|
| 57 | Persiapan run perbaikan, identitas source/runtime/pasien/fold, smoke sintetis | Selesai |
| 58 | Benchmark pengukuran LUDB 200 rekaman, deteksi dan onset/offset pada anchor referensi maupun hasil deteksi | PASS |
| 59 | Ekstraksi dan QA 230/230 pasien development, tanpa eksklusi | Selesai |
| 60 | Pilot fold 1: tiga arm fitur × (9 inner + 1 outer) = 30 fit | Selesai untuk tiga arm fitur |
| 61 | Melengkapi fold 2–5: tambahan 120 fit; analisis lima fold dan OOF | Selesai untuk tiga arm fitur |

Total **150 fit = 135 inner + 15 outer**, termasuk pilot. Masing-masing arm memiliki 50 fit, 5 pipeline outer, dan satu set OOF berisi seluruh **230 pasien unik**. Satu pasien muncul tepat sekali dalam OOF setiap arm; tiga arm bukan 690 pasien independen.

Masih tertunda: `random_l2`, `pretrained_l2`, `fusion_random`, `fusion_pretrained`. Anggaran awal seluruh tujuh arm adalah 350 fit; **200 fit bergantung pada cache** belum dijalankan. Jangan menganggap semua eksperimen V11 sudah lengkap.

## 2. Hasil utama: rata-rata macro-F1 pasien pada lima outer fold

| Input | Jumlah fitur mentah | Macro-F1 mean ± SD | Mean training F1 | Mean gap train–validasi |
|---|---:|---:|---:|---:|
| QRS + RR (`qrs_rr`) | 32 | **0,4210 ± 0,0610** | 0,5967 | 0,1756 |
| Morfologi tanpa RR (`morphology`) | 126 | **0,5555 ± 0,1070** | 0,8489 | 0,2935 |
| Morfologi + RR (`morphology_rr`) | 132 | **0,5532 ± 0,0869** | 0,8608 | 0,3076 |

SD adalah variasi antar-lima fold, bukan confidence interval. Skor 0,5555 adalah macro-F1, bukan accuracy 55,55%. Semua merupakan hasil **development CV**, belum skor test final atau validasi eksternal asal PVC.

| Fold | QRS + RR | Morfologi | Morfologi + RR | Delta tambahan RR |
|---|---:|---:|---:|---:|
| 1 | 0,4308 | 0,4914 | 0,5378 | +0,0465 |
| 2 | 0,3934 | 0,4908 | 0,4772 | −0,0136 |
| 3 | 0,5113 | 0,6904 | 0,6601 | −0,0303 |
| 4 | 0,3446 | 0,6505 | 0,6249 | −0,0256 |
| 5 | 0,4251 | 0,4542 | 0,4658 | +0,0116 |

Morfologi mengungguli QRS/RR pada **5/5 fold**, dengan kenaikan mean **0,1344 atau 13,44 poin persentase F1**. Ini perbandingan dua paket fitur, bukan estimasi manfaat satu fitur morfologi secara terisolasi: arm morfologi juga menghilangkan RR dan menggunakan lebih banyak deskriptor.

Tambahan RR pada morfologi mengubah mean sebesar **−0,0023 atau −0,23 poin persentase**, membaik hanya **2/5 fold**. Pilot fold 1 tampak lebih baik, tetapi kesimpulan itu tidak bertahan pada full CV. Morfologi tanpa RR adalah referensi fitur utama sementara, dengan selisih sangat kecil terhadap morfologi + RR; belum merupakan model final yang dikonfirmasi.

## 3. Agregasi OOF dan ketidakpastian perbandingan

Menggabungkan prediksi semua pasien terlebih dahulu menghasilkan metrik berbeda dari merata-ratakan metrik tiap fold. Keduanya dilaporkan terpisah.

| Arm | Pooled accuracy | Pooled macro-F1 | Pooled macro AUC OVR |
|---|---:|---:|---:|
| QRS + RR | 0,4304 | 0,4278 | 0,6299 |
| Morfologi | 0,5348 | 0,5562 | 0,7262 |
| Morfologi + RR | 0,5391 | 0,5548 | 0,7270 |

Paired bootstrap pasien, stratifikasi kelas, 2.000 replikasi, seed 118, menggunakan **prediksi OOF yang sudah tetap**:

| Perbandingan | Delta pooled F1 | Interval conditional 95% |
|---|---:|---:|
| Morfologi − QRS/RR | +0,1284 | [0,0564; 0,2000] |
| Morfologi + RR − Morfologi | −0,0014 | [−0,0369; 0,0330] |

Interval tambahan RR melintasi nol: tidak ada bukti konsisten dalam perbandingan ini bahwa RR meningkatkan F1. Interval ini **tidak memasukkan ketidakpastian retraining, pemilihan model, atau rangkaian eksperimen yang sudah memilih metode pada development**. Ini bukan fully nested confirmatory estimate, uji eksternal, atau bukti signifikansi klinis.

RR mengubah prediksi 19 pasien: 9 yang tadinya salah menjadi benar, 8 yang tadinya benar menjadi salah, dan 2 berubah ke kelas salah lain. Accuracy sedikit naik, tetapi macro-F1 sedikit turun karena distribusi kesalahan antarkelas berubah.

## 4. Kesalahan yang paling perlu diperbaiki

Berikut recall **pooled OOF**, berbeda sedikit dari mean recall fold:

| Arm | LVOT (57 pasien) | RVOT cusp (103) | RVOT septal/freewall/other (70) |
|---|---:|---:|---:|
| QRS + RR | 47,37% | 41,75% | 41,43% |
| Morfologi | **70,18%** | 50,49% | 44,29% |
| Morfologi + RR | 64,91% | **54,37%** | 44,29% |

Pada morfologi: **42/103** pasien cusp diprediksi other dan **37/70** pasien other diprediksi cusp. Total **79 dari 107 kesalahan** berupa pertukaran kedua kelompok RVOT. Tambahan RR mengurangi pertukaran itu menjadi 75 pasien, tetapi menurunkan jumlah LVOT yang benar dari 40 menjadi 37. Karena itu kenaikan recall cusp saja tidak cukup untuk menyatakan model membaik secara keseluruhan.

Gap morfologi sekitar **0,2935** dan morfologi + RR **0,3076** menunjukkan kemampuan pada training belum berpindah dengan baik ke pasien validasi. Fold 3 pada kedua arm mencapai training F1 **1,0000**, sedangkan validasi hanya 0,6904/0,6601. Ini konsisten dengan masalah generalisasi, bukan bukti bahwa fitur sudah sempurna atau bahwa hanya satu penyebab yang menjelaskan semua kesalahan.

Missingness bukan hambatan utama yang tampak dari kelengkapan tabel: hanya **2/230 pasien** memiliki fitur hilang, maksimum **2/132 kolom (1,52%)**, median 0, dan tidak ada pasien tanpa kandidat lengkap. Fitur lengkap tidak menjamin batas QRS, identitas beat, atau informasi lokalisasi benar. Median kandidat lengkap menurut kelas adalah 13/14/16; seluruh kandidat tetap berstatus PVC belum terverifikasi.

## 5. Pengukuran LUDB dan integritas hasil

Notebook 58: micro-F1 deteksi **0,96853**; median galat onset/offset/durasi **6/8/10 ms** pada kedua mode anchor. Coverage batas reference-anchor **97,81%**, detected-anchor **98,90%**. Gate rekayasa tetap lulus; ini bukan F1 classifier lokalisasi tiga kelompok. Referensi batas yang tidak lengkap tetap dicatat, bukan dibuatkan ground truth baru.

Audit baru membaca JSON/CSV dan menghitung ulang metrik dari prediksi tersimpan. Ia memverifikasi:

- Source/runtime/protokol dan pembagian 230 pasien tidak berubah.
- Hash seluruh 150 pipeline tersimpan beserta identitas, hasil dan prediksinya; pipeline tidak dibuka atau diinferensikan.
- Keenam metrik training/evaluation pada seluruh 150 fit cocok, termasuk gap dan teks kelas/argmax.
- Sembilan skenario C per arm/fold lengkap; seluruh 15 pilihan inner dapat direproduksi, termasuk tie-break C terkecil.
- Imputer/scaler dan class weights terikat pada ID training; tidak ada irisan pasien training/evaluation pada tiap fit.
- Manifest LUDB, sumber LUDB, manifest fitur, seluruh 230 shard pasien dan hash raw development cocok.
- Semua tabel analisis awal dan ketiga OOF dapat direproduksi.
- **3.102 berkas yang dipantau tidak berubah** saat audit. Source/backend/runner, notebook yang sudah dieksekusi, serta hasil lama tetap dipertahankan.

Tidak ada training, loading model, inferensi waveform, atau pembacaan test oleh audit ini. Perhitungan ulang metrik tidak membuktikan independensi pretraining atau validasi klinis lokal yang belum tersedia.

Audit disimpan terpisah dari analisis notebook:

`results/research_v11/20261007_morphology_regularized_features_v11_r1/analysis_20261007_complete_features/`

Kode audit: [analysis_v11/completed_features.py](../../analysis_v11/completed_features.py). Bukti utama: [audit.json](../../results/research_v11/20261007_morphology_regularized_features_v11_r1/analysis_20261007_complete_features/audit.json), [delta/CI](../../results/research_v11/20261007_morphology_regularized_features_v11_r1/analysis_20261007_complete_features/paired_bootstrap.json), dan [confusion matrices](../../results/research_v11/20261007_morphology_regularized_features_v11_r1/analysis_20261007_complete_features/confusion_matrices.json).

## 6. Tindakan berikutnya

1. **Tidak mengulang 57–61 untuk tiga arm fitur yang sudah selesai.** Simpan hasil sebagai ablation development. V11 saat ini menggunakan logistic regression L2 dengan sembilan C, belum fine-tuning CNN–Transformer baru. Jangan mengklaim bahwa fusion/neural model sudah diuji.
2. **Lengkapi kontrol encoder/fusion jika cache asli dapat dipulihkan.** Minimal perlu `patients.csv`, `random_probe.npy` dan `pretrained_probe.npy`, dengan hash yang sama dengan audit V10. Folder backup dapat ditunjuk melalui `TA_V11_FEATURE_CACHE` sebelum setup notebook; sel cache harus menyatakan ready sebelum menjalankan empat arm yang tertunda. Setelah itu lanjutkan hanya sel arm tertunda pada 60/61. Jangan menjalankan training ulang untuk tiga arm fitur. Jangan mengisi cache dengan angka nol atau mengganti representasi diam-diam.
3. **Jika backup tidak tersedia, siapkan endpoint pemulihan representasi manual dengan identitas/run yang jelas.** Jangan langsung menjalankan notebook 52 lama sebagai cara pemulihan tanpa memeriksa prasyarat cache/backend dan disk. Representasi yang berbeda dari hash audit harus menjadi eksperimen baru, bukan dicampurkan ke run `_r1`. Audit ini tidak meregenerasi encoder atau menjalankan 200 fit yang masih tertunda.
4. **Prioritas ilmiah berikut: identitas dan agregasi denyut, serta pemisahan cusp–other.** Paper asal memasukkan kompleks ketika PVC/VT terjadi dan menggunakan pilihan morfologi operator. Project sekarang mengambil median/IQR seluruh kandidat QRS tanpa label beat PVC. Karena itu pencampuran beat merupakan hipotesis yang perlu diuji, belum penyebab yang sudah terbukti. [Zheng dkk., 2022, protokol input ECG dan pembahasan](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2022.809027/full).
5. **Tanpa ahli ECG, bangun evaluasi beat dengan data beranotasi dan pemeriksaan transfer domain sebelum mengubah input lokalisasi.** Pemilihan beat terlebar atau confidence classifier tidak menjadi ground truth PVC. Metode agregasi/seleksi/dimensionalitas berikut dipilih hanya pada training/inner split, lalu dibandingkan dengan kontrol yang sama pada outer pasien. Jika fitur/fusion terbukti membantu, baru integrasikan ke CNN–Transformer melalui eksperimen tersendiri.
6. **Model final dan klaim publikasi menunggu metode stabil serta evaluasi independen.** Historical test yang sudah dilihat bukan unseen baru. Nilai 0,5555 belum terbukti maksimal, tetapi hasil saat ini juga belum mendukung janji macro-F1 0,9 atau penerimaan Q1.

Parameter C terpilih per fold:

| Arm | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| QRS + RR | 0,1 | 0,001 | 0,1 | 10 | 100 |
| Morfologi | 0,01 | 0,1 | 10 | 0,01 | 0,1 |
| Morfologi + RR | 0,01 | 0,1 | 10 | 0,01 | 0,1 |

Kelima nilai tersebut adalah pilihan inner untuk **masing-masing fold**, belum satu C final yang boleh dipilih dari fold outer terbaik.

## Grafik hasil audit

![Skor lima fold dan gap](../../results/research_v11/20261007_morphology_regularized_features_v11_r1/analysis_20261007_complete_features/fold_scores_and_gap.png)

![Confusion matrix OOF](../../results/research_v11/20261007_morphology_regularized_features_v11_r1/analysis_20261007_complete_features/pooled_oof_confusions.png)
