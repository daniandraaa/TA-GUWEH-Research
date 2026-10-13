# Fase revisi TA: fine-tuning CNN-Transformer

**Status fase V9 — 6 Oktober 2026:** notebook **46–50 selesai**, 20 fit baru + 10 referensi valid. Median epoch memberi kenaikan F1 Encoder yang disertai penurunan recall other; Hierarchical turun. [Hasil V9](ANALISIS_HASIL_V9_20261006.md). Berikutnya [audit representasi dan protokol fine-tuning pretrained](RISET_REPRESENTASI_DAN_FINE_TUNING_20261006.md). Tidak mengulang V9 atau membuka historical test untuk memilih recipe berikut.

**Fase aktif V9 — 6 Oktober 2026:** notebook **46–50** tersedia untuk median epoch inner pada real-only, dengan 10 referensi identik V8 dan 20 fit baru. Persiapan/diagnosis selesai; smoke/training manual belum dijalankan. [Urutan manual](PANDUAN_MANUAL_V9.md) · [kajian CNN–Transformer pretrained](RISET_REPRESENTASI_DAN_FINE_TUNING_20261006.md). Hasil V9 belum ada; final/test tetap belum dibuka lagi.

**Status fase — 6 Oktober 2026:** V8/notebook **41–45 selesai** dan 120 fit lolos audit. GAN memperbaiki mutu V7, tetapi belum menjadi peningkatan classifier yang konsisten; kedua model kehilangan recall RVOT other. [Hasil V8](ANALISIS_HASIL_V8_20261006.md). Berikutnya stabilitas seleksi epoch serta representasi/riset pretraining real-only pada run baru; [urutan fase](FASE_SETELAH_V8_20261006.md). Kode/runtime/checkpoint V4–V8 dipertahankan; tidak ada final/test baru.

**Fase aktif V8 — 5 Oktober 2026:** implementasi classifier K4 real-only/real augmentation/GAN augmentation tersedia. Jalankan **41 → 42** dahulu, lalu **43–45**. Semua arm menggunakan epoch inner dan anggaran update sebanding; total 120 fit termasuk pilot. [Panduan](PANDUAN_MANUAL_V8.md). Hasil V7 tetap dibekukan; training classifier data asli belum dimulai.

**Fase aktif — 5 Oktober 2026:** V4, audit 28, V5 dan GAN/QC V7 selesai. K4 tetap; notebook 37–40 terverifikasi, lima fold PASS setelah hitung ulang QC native. [Hasil V7](ANALISIS_HASIL_V7_20261005.md). Berikutnya siapkan protokol/endpoint classifier pada run baru: real-only, kontrol augmentasi real dan real+GAN, lalu periksa macro-F1 pasien, kedua RVOT, variasi seed serta biaya. [Rancangan fase berikut](FASE_CLASSIFIER_SETELAH_V7_20261005.md). Belum ada skor classifier baru atau bukti manfaat klinis sintetis.

> **V4 selesai — 3 Oktober 2026:** notebook 20–27 tanpa error tersimpan; diagnosis 45/45, CV 450/450, stability 75/75, final 25/25 dan test 25/25 telah diaudit. Selanjutnya audit representasi pada kasus development yang konsisten salah, ablation terkontrol pada run baru, lalu komponen GAN/evaluasi/naskah yang belum selesai. [Hasil V4 dan urutan tindak lanjut](ANALISIS_HASIL_V4_20261003.md). Rencana/hasil sebelumnya di bawah tetap sebagai riwayat.

> **Status aktif 2 Oktober 2026 setelah V3:** 630 CV, 20 final/test, SHAP primer dan benchmark INCART 75 rekaman sudah selesai. Urutan berikutnya adalah **diagnostik/stabilitas training development → GAN terkoreksi → baseline/tuning/GAN yang sebanding → evaluasi diperkuat → naskah**. [Fase lanjutan yang konkret](FASE_LANJUTAN_SETELAH_V3_20261002.md) · [checklist terbaru](STATUS_PENYELESAIAN_DAN_LANGKAH_BERIKUTNYA.md). Bagian di bawah mempertahankan riwayat V1/V2; pernyataan lama seperti SHAP V2 belum dibuat bukan status terkini.

> **Sesudah notebook 10:** seluruh 7 sel analisis berhasil dijalankan tanpa error. Status pekerjaan V2 dan tugas yang masih terbuka dicatat pada [checklist aktif penyelesaian](STATUS_PENYELESAIAN_DAN_LANGKAH_BERIKUTNYA.md). SHAP historis pada checklist di bawah merujuk pada run V1; **SHAP final V2 belum dibuat**.


> **Hasil aktual 1 Oktober 2026:** seluruh 270 CV, 15 model final, dan 15 evaluasi test V2 telah selesai. Encoder tetap model utama dari CV; test macro-F1 seed 42 = 0,635, mean lima seed = 0,616 ± 0,034. Analisis menemukan 25 pasien development salah pada ketiga model serta kebingungan antara kelompok RVOT. Lihat [analisis lengkap](ANALISIS_HASIL_V2_20261001.md) dan notebook `10_Analisis_Hasil_V2.ipynb`. Prioritas berikutnya adalah review/kurasi PVC, kontribusi training antar-pasien, dan pengujian terkontrol; GAN baru/validasi eksternal belum selesai.


> **Pembaruan 30 September 2026 — prioritas V2:** sumber dataset telah diperiksa; sampling publik adalah **2.000 Hz**. Pengguna menyetujui koreksi tiga kelompok menjadi **LVOT / RVOT cusp / RVOT septal-free-wall-other**, dengan anggota pasien yang sama. Label historis “Non-OT” (LC/AC/RC) tidak sesuai anatomi sumber. Angka lama dipertahankan sebagai hasil historis. Mulai pekerjaan baru dari notebook **06**, mengikuti [panduan V2](PANDUAN_REVISI_V2.md). Training V2 belum dijalankan; GAN baru masih tahap berikutnya.


Tanggal audit dan awal eksekusi revisi: 29 September 2026; audit hasil manual diperbarui 30 September 2026. Dokumen ini disusun dari proposal, kode, keluaran notebook, dan artefak hasil di folder project. Bagian 2–3 mencatat eksperimen historis; status implementasi revisi dicatat pada fase dan checklist. Training revisi menggunakan runner modular dengan run ID `20260929_grid9_v1`. Setelah proses otomatis dihentikan, pengguna menyelesaikan notebook 00–04 secara manual. Hasilnya dibahas lebih lanjut pada [analisis eksperimen](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/ANALISIS_HASIL_EKSPERIMEN.md>).

Jumlah artefak lengkap tersedia pada [status.json](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/status.json>): fine-tuning 135/135 CV, baseline 15/15 CV, enam model final, dan enam evaluasi test. [progress.json](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/progress.json>) masih mencatat penghentian runner otomatis lama; field itu tidak mewakili hasil eksekusi manual kemudian. Panduan eksekusi tersedia pada [README revisi](</Users/daniandraaa/Desktop/TA GUWEH/README_REVISI.md>); grafik baru dibuat dari artefak yang sama oleh [notebook 05](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>).

**Target revisi:** mencari konfigurasi learning rate dan batch size terbaik melalui 5-fold CV pada data pengembangan 80%, melatih ulang model dengan konfigurasi terpilih, lalu melakukan inferensi pada hold-out 20% tanpa CV pada test.

**Grid revisi yang disepakati:** learning rate `[1e-4, 5e-4, 1e-3]` dan batch size `[16, 32, 64]`. Seluruh sembilan kombinasi digunakan pada ketiga arsitektur dengan lima fold yang sama.

**Makna model final:** satu model hasil retraining untuk setiap arsitektur, bukan lima model fold yang dijadikan model final. Proposal membandingkan tiga arsitektur, sehingga keluaran penelitian dapat tetap berisi tiga checkpoint final. Jika dibutuhkan satu model utama untuk digunakan, pilih arsitekturnya berdasarkan hasil CV sebelum melihat hasil test.

## 1. Acuan proposal dan arahan bimbingan

| Acuan | Isi yang menjadi dasar revisi |
|---|---|
| Proposal, bagian 1.4, halaman tercetak 3 / halaman PDF 8 | Mengembangkan tiga model, melakukan fine-tuning, serta membandingkan baseline dan hasil fine-tuning. |
| Proposal, bagian 1.5, halaman tercetak 4 / halaman PDF 9 | Fine-tuning berfokus pada hiperparameter tanpa mengubah struktur dasar arsitektur; GAN hanya pada skenario fine-tuning. |
| Proposal, Gambar 3.3, halaman tercetak 33 / halaman PDF 38 | Hold-out 80/20, pencarian parameter dengan CV, retraining, inferensi akhir, dan SHAP. |
| Proposal, uraian objektif kedua, halaman tercetak 34-39 / halaman PDF 39-44 | Konfigurasi terbaik ditentukan dari performa validasi antar-fold; test 20% tidak digunakan untuk pemilihan parameter. |
| Proposal, Tabel 3.3, halaman tercetak 50 / halaman PDF 55 | Learning rate: `1e-3`, `5e-4`, `1e-4`; batch size: `64`, `128`, `256`. |
| Arahan bimbingan terbaru | Batasi pencarian pada 3 learning rate × 3 batch size; setiap kombinasi menjalani 5 fold; pecah proses agar dapat dijalankan per tahap atau per model. |
| Keputusan revisi yang disepakati | Learning rate `[1e-4, 5e-4, 1e-3]`; batch size `[16, 32, 64]`. Nilai learning rate tetap sama dengan proposal, diurutkan dari kecil ke besar; batch size disesuaikan untuk eksperimen revisi. |

Nilai kandidat aktif dalam dokumen ini mengikuti keputusan revisi yang disepakati dan telah diterapkan pada `configs/scenarios.yaml`. Kode historis `03_Finetune_CNN_Transformer.py` tetap menjadi bukti eksperimen lama. Perubahan batch size serta metodologi GAN dijelaskan pada [draf revisi Bab III](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/REVISI_METODOLOGI_BAB_III.md>), untuk digunakan memperbarui proposal. Parameter terbaik kini tersedia dari grid CV lengkap dan dicatat pada [analisis hasil](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/ANALISIS_HASIL_EKSPERIMEN.md>).

Proposal perlu dirapikan pada bagian GAN: Gambar 3.3 dan uraian fold menyebut validasi asli, sedangkan uraian skenario dan baris validasi Tabel 3.3 menyebut validasi data hasil GAN. Revisi yang diusulkan menggunakan **data asli untuk validation dan test; GAN dilatih hanya dari bagian training yang sedang digunakan**. Saat retraining final, GAN boleh dilatih dari seluruh 80% data pengembangan. Perubahan penempatan GAN ini perlu dijelaskan dalam revisi metodologi, karena diagram lama menempatkannya sebelum CV.

Pengertian fine-tuning dalam proposal ini adalah optimasi hiperparameter dan retraining pada arsitektur yang sama. Kode saat ini membuat model baru untuk tiap fold dan retraining final; tidak ditemukan proses memuat checkpoint baseline untuk melanjutkan bobotnya. Melanjutkan bobot baseline yang sudah dilatih pada seluruh 80% ke setiap fold justru akan membuat model telah melihat pasien validation fold.

## 2. Pekerjaan yang sudah dituntaskan dan buktinya

| Pekerjaan | Bukti di project | Status terhadap revisi |
|---|---|---|
| Mengumpulkan data dan metadata diagnosis | 334 CSV di masing-masing folder `PVCVTRawECGData`, `PVCVTECGData`, serta `Diagnosis.xlsx` | Data tersedia. |
| EDA dan implementasi preprocessing | `01_EDA_Preprocessing.py`, notebook EDA, grafik distribusi, missing value, statistik, dan PSD | Sudah dikerjakan; tidak perlu memulai EDA dari awal. |
| Menghasilkan data hasil DWT | 334 CSV pada `preprocessed_data/dwt_denoised`; kode HPF 0,5 Hz dan DWT `coif5`, level 5, threshold SURE | Artefak preprocessing tersedia. |
| Membentuk kelas dan memisahkan data utama | Keluaran notebook: 289 pasien utama, terdiri dari LVOT 71, Non-OT 124, RVOT 94; Summit 5 dan unknown sublocation 40 terpisah | Sudah dikerjakan. Unknown tidak mempunyai label sublokasi untuk evaluasi akurasi terawasi. |
| Menyimpan pembagian pasien train/test | `train_ids.npy`, `test_ids.npy`, `split_metadata.csv`; irisan ID train/test yang diperiksa kosong | Pembagian berbasis pasien sudah tersedia. Split awal 267/67 pasien; setelah seleksi tiga kelas menjadi 230/59. |
| Implementasi CNN-Transformer Encoder, CNN-PatchTST, dan CNN-Hierarchical Transformer | Definisi model pada `02_Baseline_CNN_Transformer.py` dan `03_Finetune_CNN_Transformer.py` | Tiga model sudah ada; struktur baseline dan fine-tuning perlu disamakan. |
| Menjalankan baseline tiga model | `baseline_results/kfold_comparison.csv`, `final_test_results.csv`, dan tiga `final_model_*.pth`; 25 sel kode notebook baseline memiliki execution count | Eksperimen baseline sudah menghasilkan keluaran. CV baseline perlu diperbarui agar scaler dan metriknya konsisten dengan revisi. |
| Menjalankan 1D-CGAN dan membuat data sintetis | Keluaran notebook fine-tuning: GAN 50 epoch dan 675 segmen sintetis; grafik loss serta real-vs-synthetic | Implementasi sudah berjalan. Penempatannya harus direvisi menjadi per training fold. |
| Menjalankan pencarian konfigurasi dengan CV | Notebook fine-tuning mencatat 4 konfigurasi × 5 fold × 3 model = **60 pelatihan** | Sudah menjalankan eksperimen pendahuluan; belum grid lengkap 9 konfigurasi yang diminta. |
| Memilih konfigurasi eksperimen lama | `finetune_results/best_configs.json`; keluaran notebook berisi skor CV dan median best epoch | Hasil pemilihan lama tersedia; belum parameter terbaik untuk grid revisi. |
| Retraining dan inferensi hold-out tiga model | Tiga `finetuned_*.pth`, `final_test_results.csv`, grafik training/inferensi | Sudah berjalan tanpa k-fold pada test. Aturan epoch dan pemilihan model utama perlu diperbaiki. |
| Analisis patient-level dan sublokasi | `evaluate_patient_wise`, `per_sublocation_test_results.csv`, grafik confidence dan sublokasi | Sudah dikerjakan. Agregasi probabilitas segmen per pasien dapat digunakan kembali. |
| SHAP dan importance 12 sadapan | Keluaran notebook menyatakan GradientExplainer berhasil; `lead_importance.csv`, `shap_summary_beeswarm.png` | Hasil interpretasi eksperimen lama tersedia; perlu dihasilkan kembali untuk model revisi. |

Nama relatif pada tabel di atas merujuk ke folder project yang sama dengan dokumen ini. Lokasi beberapa bukti utama:

- [Kode EDA dan preprocessing](</Users/daniandraaa/Desktop/TA GUWEH/01_EDA_Preprocessing.py>)
- [Kode baseline](</Users/daniandraaa/Desktop/TA GUWEH/02_Baseline_CNN_Transformer.py>)
- [Kode fine-tuning historis](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py>)
- [Notebook fine-tuning beserta keluaran eksperimen](</Users/daniandraaa/Desktop/TA GUWEH/arsip/notebooks_lama/03_Finetune_CNN_Transformer.ipynb>)
- [Hasil CV baseline](</Users/daniandraaa/Desktop/TA GUWEH/baseline_results/kfold_comparison.csv>)
- [Konfigurasi terbaik eksperimen lama](</Users/daniandraaa/Desktop/TA GUWEH/finetune_results/best_configs.json>)
- [Hasil inferensi fine-tuning lama](</Users/daniandraaa/Desktop/TA GUWEH/finetune_results/final_test_results.csv>)

Hasil patient-level yang telah tersimpan, dibulatkan empat desimal:

| Model | F1 macro baseline | F1 macro fine-tuning lama | Accuracy fine-tuning lama | AUC macro fine-tuning lama |
|---|---:|---:|---:|---:|
| CNN-Transformer Encoder | 0,4778 | 0,5981 | 0,5763 | 0,7428 |
| CNN-PatchTST | 0,4162 | 0,6200 | 0,6102 | 0,7917 |
| CNN-Hierarchical Transformer | 0,5837 | 0,6102 | 0,5932 | 0,8030 |

Tabel ini mencatat hasil yang sudah ada. Angkanya belum membuktikan pengaruh hiperparameter saja, karena arsitektur serta sebagian aturan training baseline dan fine-tuning berbeda, dan prosedur CV perlu diperbaiki.

Konfigurasi lama yang terpilih:

| Model | Learning rate | Batch size | Dropout Transformer | Weight decay | Mean CV F1 dari log, sekitar | Median best epoch | Epoch retraining aktual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Encoder | `2e-4` | 32 | 0,3 | `1e-4` | 0,650 | 5 | 8 |
| PatchTST | `2e-4` | 32 | 0,3 | `1e-4` | 0,665 | 6 | 8 |
| Hierarchical | `1e-4` | 16 | 0,3 | `1e-4` | 0,663 | 6 | 8 |

## 3. Selisih implementasi historis dengan protokol revisi

| Temuan | Lokasi bukti | Perbaikan yang diperlukan |
|---|---|---|
| Daftar grid membentuk 36 kombinasi, tetapi loop training menggunakan empat `subset_params` | [Kode pencarian](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:865>) | Gunakan seluruh 9 kombinasi learning rate × batch size; parameter lain dibuat tetap. |
| Scaler di-fit pada seluruh 80% sebelum CV | [Fine-tuning](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:253>), [baseline](</Users/daniandraaa/Desktop/TA GUWEH/02_Baseline_CNN_Transformer.py:234>) | Di CV, fit scaler hanya pada training fold lalu transform validation fold. Untuk final, fit ulang pada seluruh 80%. |
| GAN dilatih dari seluruh 80% lalu satu kumpulan sintetis ditambahkan ke setiap training fold | [Dataset GAN](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:408>), [loop CV](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:916>) | Latih GAN hanya pada training fold terkait. Validation harus tetap data asli. |
| Struktur model berubah dibanding baseline | [AttentionPooling](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:617>) dan [Hierarchical](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:693>) | Gunakan satu definisi arsitektur bersama. Acuan revisi ini adalah struktur baseline; tambahan attention pooling/fusion sekarang tidak diperlakukan sebagai perubahan hiperparameter. |
| CV baseline dinilai per segmen, sedangkan CV fine-tuning dinilai per pasien | [Evaluasi baseline](</Users/daniandraaa/Desktop/TA GUWEH/02_Baseline_CNN_Transformer.py:545>), [evaluasi fine-tuning](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:811>) | Gunakan metrik utama patient-level untuk keduanya, dengan agregasi probabilitas yang sama. Bila tetap melaporkan segment-level, pisahkan tabelnya. |
| Model utama dipilih memakai F1 test | [Fine-tuning](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:1237>), [baseline](</Users/daniandraaa/Desktop/TA GUWEH/02_Baseline_CNN_Transformer.py:796>) | Pilih konfigurasi dan arsitektur utama dari CV sebelum evaluasi test. Hasil test boleh dilaporkan untuk ketiga model yang sudah ditetapkan. |
| Median best epoch diberi batas minimum 8 | [Retraining](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:1086>) | Gunakan aturan epoch yang ditetapkan sebelum test, misalnya median best epoch apa adanya; jangan mengubahnya berdasarkan skor test. |
| JSON akhir hanya menyimpan empat parameter | [Penyimpanan](</Users/daniandraaa/Desktop/TA GUWEH/03_Finetune_CNN_Transformer.py:1566>) | Simpan hasil setiap fold, rerata/SD, epoch, seed, konfigurasi lengkap, scaler, label mapping, dan identitas eksperimen. |
| Kode memakai 1.000 Hz, sementara satu komentar menyebut segmen 5.000 sampel sebagai 10 detik pada 500 Hz | Konfigurasi preprocessing dan komentar segmentasi fine-tuning | Catat sumber sampling rate. Pada 1.000 Hz, 5.000 sampel = 5 detik. Perbaiki narasi setelah verifikasi; jangan mengganti frekuensi data hanya karena komentar. |

Test 59 pasien sudah pernah dievaluasi dan digunakan dalam pemilihan `best_final`. Karena itu, hasil lama harus dicatat sebagai eksperimen terdahulu. Membekukan protokol berikutnya tidak membuat test historis kembali menjadi data yang belum pernah dilihat. Untuk klaim evaluasi benar-benar baru, diperlukan data pasien yang belum pernah dipakai; mengacak ulang pasien yang sama tidak menyelesaikan masalah tersebut. Revisi tetap dapat mempertahankan hold-out historis untuk perbandingan yang transparan, sambil menyatakan keterbatasannya di laporan.

## 4. Sembilan skenario parameter yang telah diuji

Grid yang disepakati untuk ketiga model:

```python
grid_params = {
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'batch_size': [16, 32, 64],
}
```

| Skenario | Learning rate awal | Batch size | Fold |
|---|---:|---:|---|
| S01 | `1e-4` | 16 | F1-F5 |
| S02 | `1e-4` | 32 | F1-F5 |
| S03 | `1e-4` | 64 | F1-F5 |
| S04 | `5e-4` | 16 | F1-F5 |
| S05 | `5e-4` | 32 | F1-F5 |
| S06 | `5e-4` | 64 | F1-F5 |
| S07 | `1e-3` | 16 | F1-F5 |
| S08 | `1e-3` | 32 | F1-F5 |
| S09 | `1e-3` | 64 | F1-F5 |

Dasar penetapan grid:

- Learning rate mencakup rentang 10 kali, dari `1e-4` hingga `1e-3`, untuk membandingkan langkah pembelajaran kecil, menengah, dan besar dalam ruang pencarian ini.
- Batch size 16, 32, dan 64 sesuai dengan rentang eksperimen awal yang sudah dijalankan. Konfigurasi lama memilih batch 32 untuk Encoder/PatchTST dan batch 16 untuk Hierarchical; hasil itu menjadi petunjuk awal, bukan bukti bahwa batch tersebut pasti optimal, karena parameter lain ikut berubah dan prosedur CV masih perlu diperbaiki.
- Setiap input berukuran 12 sadapan × 5.000 sampel. Rentang batch yang disepakati mengurangi kebutuhan memori dibanding rentang batch dalam proposal awal. Pemilihan parameter terbaik tetap berdasarkan hasil CV lengkap.

**Setiap skenario harus menjalani lima fold yang sama.** Jangan menaruh satu konfigurasi berbeda pada setiap fold, karena pengaruh konfigurasi akan tercampur dengan perbedaan pasien.

- Per model: 9 skenario × 5 fold = **45 pelatihan CV**.
- Tiga model: 3 × 45 = **135 pelatihan CV**.
- Retraining final standar: **3 pelatihan tambahan**, satu per model, tanpa CV pada test.
- Angka di atas tidak memasukkan pelatihan GAN, baseline ulang, ataupun pengulangan seed opsional.

Pada setiap fold, sekitar 4/5 dari pool 80% menjadi training dan 1/5 menjadi validation. Secara nominal ini setara dengan sekitar 64% dan 16% dari dataset utama; hold-out 20% tetap terpisah. Jumlah pasien aktual mengikuti pengelompokan pasien dan pembulatan.

Parameter tetap yang dibekukan pada `configs/finetune_fixed.yaml` agar ruang pencarian hanya sembilan skenario:

| Parameter | Konfigurasi tetap untuk revisi | Dasar |
|---|---|---|
| Optimizer | Adam | Tabel 3.3; kode fine-tuning historis memakai AdamW, sehingga perbedaannya dicatat pada draf metodologi. |
| Weight decay | `1e-4` | Salah satu opsi proposal. |
| Maksimum epoch CV | 100 | Salah satu opsi proposal. Early stopping tetap berjalan. |
| Early stopping patience | 10 | Salah satu opsi proposal. |
| Scheduler | Cosine Annealing, `T_max=100`, minimum LR `1e-6` | Opsi proposal; pertahankan kebijakan scheduler ketika retraining. |
| Gradient clipping | 1,0 | Salah satu opsi proposal. |
| Loss | Cross-entropy tanpa class weighting dan tanpa label smoothing | Opsi proposal; kelas training diseimbangkan melalui GAN. |
| Arsitektur, dropout, segmentasi, overlap | Satu definisi tetap yang sama untuk perbandingan | Struktur baseline menjadi acuan. |
| GAN | Konfigurasi tetap, training per fold | Efek learning rate dan batch size model klasifikasi dapat dibandingkan pada augmentasi yang sama. |
| Metrik pemilihan | Mean patient-level macro-F1 pada 5 fold | Ketiga kelas berkontribusi terhadap skor pemilihan. |

Parameter tetap dan kandidat grid telah diimplementasikan. Arsitektur ketiga model telah diverifikasi kompatibel dengan checkpoint baseline historis dan lolos forward/backward aktual di GPU Metal. Grid CV lengkap menghasilkan S09 untuk Encoder, S02 untuk PatchTST, dan S06 untuk Hierarchical. Konfigurasi, fingerprint kode/input, serta versi lingkungan dibekukan dalam `protocol.json`; runner menolak pencampuran hasil jika identitas berubah.

## 5. Fase penyelesaian revisi

### Fase 1 - Bekukan protokol eksperimen

**Pekerjaan:** dokumentasikan sembilan skenario dari learning rate `[1e-4, 5e-4, 1e-3]` × batch size `[16, 32, 64]` yang sudah disepakati. Bekukan parameter tetap, target kelas, definisi arsitektur bersama, metrik utama, aturan pemilihan konfigurasi, aturan epoch final, dan seed. Pertahankan ID hold-out yang ada untuk perbandingan historis. Perbarui batch size pada Tabel 3.3, serta perbaiki uraian GAN dan fokus tuning pada metodologi proposal.

**Keluaran:** konfigurasi eksperimen dan catatan protokol. Model utama ditentukan sebelum evaluasi test. Untuk skor yang sama, gunakan aturan pemecah seri yang ditetapkan sebelumnya, misalnya SD macro-F1 yang lebih rendah, lalu urutan ID skenario.

**Syarat selesai:** daftar kandidat berisi tepat 9 konfigurasi; parameter lain tetap; narasi proposal, kode yang direncanakan, dan aturan evaluasi tidak bertentangan.

**Status:** protokol software dan konfigurasi sudah dibekukan, grid sudah diterapkan, dan draf revisi metodologi tersedia. Proposal PDF asli belum diedit; source editable tidak ditemukan di project.

### Fase 2 - Pecah kode menjadi proses yang bisa dijalankan terpisah

**Pekerjaan:** pindahkan pemuatan data, preprocessing, model, GAN, training, metrik, dan penyimpanan ke modul bersama. Buat runner yang menerima pilihan model, skenario, dan fold. Notebook per model memanggil modul yang sama.

**Keluaran:** struktur pada bagian 6; proses persiapan, tuning, retraining, inferensi, SHAP, dan perbandingan dapat dijalankan sendiri-sendiri. Mengimpor definisi model tidak langsung memulai training.

**Syarat selesai:** CNN-Transformer Encoder dapat dijalankan sendiri; demikian juga PatchTST dan Hierarchical. Satu skenario/fold dapat dilanjutkan tanpa mengulang hasil yang sudah lengkap. Identitas konfigurasi harus diperiksa sebelum memakai hasil cache.

**Status:** selesai diimplementasikan dalam `src/`, `scripts/`, dan enam notebook berurutan pada `notebooks/`: persiapan, tiga tuning per model, final/evaluasi, lalu visualisasi. Enam pengujian runner lulus, termasuk resume setelah interupsi, guard seleksi sebelum test, dan pemeriksaan seluruh grid pada fixture kecil. Pengguna menjalankan notebook 00–04 tanpa error tersimpan; notebook 05 menghasilkan 19 grafik tanpa training. Panduan tersedia pada [README revisi](</Users/daniandraaa/Desktop/TA GUWEH/README_REVISI.md>).

### Fase 3 - Siapkan lima fold beserta preprocessing dan GAN yang terpisah

**Pekerjaan:** buat manifest fold dari 230 pasien training utama dengan stratifikasi tiga kelas dan pengelompokan `HospitalID`. Semua segmen satu pasien berada dalam satu sisi split. Fit scaler pada training fold. Latih GAN dari training fold, lalu tambahkan data sintetis hanya ke training fold. Validation tetap data asli yang ditransformasi dengan scaler training fold.

**Keluaran:** lima manifest fold, lima scaler, serta checkpoint/hasil sintetis GAN yang mencatat asal training fold. Karena konfigurasi GAN tetap, hasil GAN satu fold dapat dipakai kembali oleh sembilan skenario dan tiga model pada fold yang sama. GAN final kemudian dilatih ulang pada seluruh 80%.

**Syarat selesai:** tidak ada irisan pasien train/validation/test; setiap pasien development muncul sebagai validation tepat sekali; tiga kelas tercakup di fold sejauh komposisi data memungkinkan; scaler dan GAN tidak belajar dari validation.

**Status:** selesai. 230 pasien / 2.271 segmen development disiapkan; manifest lima fold, lima scaler training fold, serta GAN pada kelima fold dan seluruh development final selesai. Audit ID menegaskan tidak ada irisan train/validation/test; scaler dan GAN tiap fold hanya memakai training fold. GAN final menambahkan 675 segmen sintetis sehingga setiap kelas training final memiliki 982 segmen. Audit tambahan pada semua fold, GAN final, dan 12 sadapan menemukan perubahan antar-sampel sintetis 10,3–54,7× real; hubungan sadapan ekstremitas juga jauh kurang konsisten. Kualitas GAN masih perlu dievaluasi sebelum diberi klaim realistis.

### Fase 4 - Jalankan seluruh skenario per arsitektur

**Pekerjaan:** periksa satu skenario/fold terlebih dahulu untuk memastikan bentuk tensor, pemakaian memori, label, dan penyimpanan hasil. Lalu selesaikan 45 training Encoder, 45 PatchTST, dan 45 Hierarchical. Urutan per model membantu pemantauan dan melanjutkan proses.

Untuk setiap training, catat loss, metrik per epoch, best epoch, durasi, seed, dan konfigurasi. Simpan checkpoint terbaik berdasarkan validation macro-F1 dan evaluasi ulang checkpoint tersebut untuk menyimpan seluruh metrik pada epoch yang sama. Reset bobot serta RNG sesuai aturan seed pada setiap training.

**Keluaran:** 45 baris hasil fold dan 9 baris ringkasan skenario per model. Ringkasan mencakup mean dan SD accuracy, sensitivity macro, specificity macro, F1 macro, AUC macro, serta median best epoch. Simpan prediksi validation per pasien untuk pemeriksaan hasil.

**Syarat selesai:** seluruh 135 pasangan model-skenario-fold mempunyai hasil lengkap; hasil gagal tidak dihitung sebagai skor nol atau diabaikan diam-diam; konfigurasi terbaik berasal dari semua lima fold, bukan satu fold tertinggi.

**Status:** selesai. Ketiga notebook tuning masing-masing memiliki 45/45 hasil CV (total 135/135), sembilan ringkasan skenario per model, history, checkpoint, dan prediksi validation. Mean/SD pada ringkasan cocok dengan lima fold sumber. Hasil 60 training historis tetap terpisah karena protokolnya berbeda.

### Fase 5 - Pilih parameter terbaik dan latih model final

**Pekerjaan:** pilih satu skenario terbaik per arsitektur dari mean patient-level macro-F1 lima fold. Tentukan arsitektur utama dari hasil CV tersebut. Gunakan median best epoch dari konfigurasi terpilih sebagai durasi final yang dibekukan. Fit scaler final pada seluruh 230 pasien development, latih GAN final pada data itu, lalu retrain model dengan data asli dan sintetis development.

Model final dibuat dari inisialisasi baru sesuai seed yang ditetapkan. Checkpoint fold berfungsi sebagai bukti validasi; model final memperoleh seluruh data development. Pertahankan kebijakan scheduler yang telah ditetapkan di CV agar perubahan durasi tidak sekaligus mengubah kurva LR tanpa penjelasan.

**Keluaran:** satu checkpoint final per arsitektur, scaler, label mapping, konfigurasi lengkap, skor CV pemilihan, median epoch, seed, dan manifest eksperimen. Tiga model boleh disimpan untuk kebutuhan pembandingan proposal; satu diberi status model utama dari hasil CV.

**Syarat selesai:** retraining memakai konfigurasi terpilih, tanpa penyesuaian dari test; artefak dapat dimuat kembali untuk inferensi tanpa menjalankan ulang CV.

**Status:** selesai. Encoder memilih S09 (`1e-3`, batch 64, 24 epoch), PatchTST S02 (`1e-4`, batch 32, 6 epoch), dan Hierarchical S06 (`5e-4`, batch 64, 8 epoch). Encoder ditetapkan sebagai model utama dari mean macro-F1 CV. Model final revisi dan scaler/GAN final tersimpan dengan seed 42.

### Fase 6 - Inferensi hold-out 20% dan pengulangan opsional

**Pekerjaan:** muat checkpoint final serta scaler development, lalu lakukan inferensi pada 59 pasien hold-out. Rata-ratakan probabilitas seluruh segmen milik seorang pasien, kemudian ambil kelas dengan probabilitas tertinggi. Laporkan confusion matrix, classification report, accuracy, sensitivity, specificity, macro-F1, dan macro AUC-ROC.

**Keluaran:** prediksi/probabilitas per pasien dan tabel metrik final untuk model yang sudah ditetapkan sebelum test. Pengujian ini tidak membagi test menjadi fold dan tidak memperbarui bobot.

**Jika ingin diulang lima kali:** lakukan lima retraining independen dengan konfigurasi sama dan seed yang ditetapkan sebelumnya, misalnya `42, 43, 44, 45, 46`, memakai pembagian development/test yang sama. Laporkan mean ± SD metrik test untuk menggambarkan variasi training. Tetapkan sejak awal checkpoint utama, misalnya seed 42; jangan memilih seed dengan skor test tertinggi. Jika pemilihan seed memang diperlukan, gunakan prosedur validation di development sebelum test.

Menjalankan inferensi deterministik lima kali pada bobot dan data yang sama tidak menghasilkan lima eksperimen independen. Merata-ratakan prediksi lima model menjadi ensemble adalah rancangan lain dan tidak diperlukan untuk target satu model final.

**Syarat selesai:** tidak ada k-fold pada test, tidak ada pemilihan konfigurasi/epoch/seed/model berdasarkan test, dan riwayat pemakaian hold-out lama dijelaskan. Pengulangan seed bersifat opsional; tiga final training menjadi 15 jika ketiga arsitektur masing-masing diulang lima kali.

**Status:** inferensi revisi selesai untuk tiga fine-tuning dan tiga baseline pada 59 pasien yang sama, tanpa fold pada test. Test macro-F1 fine-tuning: Encoder 0,6367, PatchTST 0,6775, Hierarchical 0,5270. Hasil ini memakai satu seed final (42); lima retraining independen belum dijalankan dan tidak diwajibkan oleh protokol aktif.

### Fase 7 - Samakan pembanding baseline, perbarui SHAP, dan tulis hasil

**Pekerjaan:** perbarui CV baseline dengan manifest pasien yang sama, scaler per training fold, serta metrik patient-level yang sama; jika class weights dihitung dari data, hitung dari training fold terkait. Baseline tetap tanpa GAN dan tanpa grid search. Pertahankan struktur model yang sama dengan fine-tuning.

Jalankan SHAP untuk model revisi dengan background data development. Sajikan kontribusi sadapan dan contoh kontribusi pada waktu sinyal bila dibutuhkan untuk pembahasan representasi DWT. Gunakan model utama yang sudah dipilih dari CV. Analisis sublokasi boleh dilaporkan dari prediksi kelas utama; itu bukan bukti bahwa model mengklasifikasikan sublokasi secara langsung. Summit dan unknown tetap analisis tambahan di luar akurasi tiga kelas utama.

Susun tabel baseline-versus-fine-tuning pada **tingkat evaluasi yang sama**: CV patient-level dibanding CV patient-level; hold-out patient-level dibanding hold-out patient-level. Jelaskan bahwa skenario fine-tuning mencakup GAN dan optimasi parameter, sehingga selisih hasil tidak mengisolasi pengaruh learning rate saja. Peningkatan kinerja harus mengikuti hasil aktual, bukan menjadi syarat agar eksperimen dianggap berhasil.

**Keluaran:** tabel hasil grid, parameter terpilih, hasil final, perbandingan baseline, grafik, SHAP, serta revisi Bab III dan bahan Bab IV/V.

**Syarat selesai:** seluruh angka laporan dapat ditelusuri ke artefak eksperimen; perbedaan protokol lama dan baru tercatat; keterbatasan hold-out historis serta jumlah pasien per kelas dijelaskan.

**Status:** baseline CV 15/15, baseline final/test tiga model, SHAP model utama fine-tuning dan baseline, tabel perbandingan, serta 19 grafik visualisasi sudah tersedia. Fine-tuning Hierarchical menurun pada test dibanding baseline, sehingga hasil negatif ini perlu dibahas apa adanya. Sisa pekerjaan utama adalah menerapkan draf Bab III ke naskah editable, menyusun Bab IV/V, mengevaluasi kualitas sinyal GAN, dan mencatat keterbatasan hold-out historis. Inferensi tambahan Summit/unknown belum dilakukan pada alur manual.

## 6. Rancangan struktur file dan titik menjalankan proses

Struktur berikut sudah diimplementasikan. "Endpoint" di sini berarti titik menjalankan suatu tahap melalui CLI atau notebook. Alur training dimulai dari notebook 00–04 dan sudah selesai; langkah sekarang adalah [notebook 05](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>). Panduan perintah lengkap tersedia pada `README_REVISI.md`, dan peta folder ada pada `docs/PETA_FILE.md`.

```text
configs/
  finetune_fixed.yaml             # parameter yang tidak dicari
  scenarios.yaml                  # tepat sembilan pasangan LR dan batch size
src/
  data.py                         # sinyal, label, segmentasi
  splits.py                       # hold-out dan manifest lima fold
  preprocessing.py                # scaler training fold / development final
  augmentation.py                 # GAN per fold dan GAN final
  gan_models.py                   # generator dan discriminator
  training.py                     # training, early stopping, checkpoint
  evaluation.py                   # metrik dan agregasi patient-level
  artifacts.py                    # konfigurasi, manifest, penyimpanan/resume
  cli.py                          # orchestration dan runner per tahap
  reports.py                      # status, tabel, grafik, laporan
  explain.py                      # SHAP dan prediksi supplementary
  models/
    cnn.py
    transformer_encoder.py
    patchtst.py
    hierarchical_transformer.py
scripts/
  00_prepare_data.py
  01_prepare_folds.py
  02_prepare_fold_augmentation.py
  03_tune.py                      # --model, --scenario, --fold
  04_select_best.py
  05_train_final.py               # --model, --seed
  06_evaluate_test.py             # hanya inference
  07_explain.py
  08_compare.py
  run_revision.py                 # seluruh pipeline atau satu tahap
  verify_revision.py              # kompatibilitas model dan uji GPU
notebooks/
  00_Mulai_di_Sini.ipynb            # persiapan dan GAN bersama
  01_Tuning_Encoder.ipynb            # 45 kasus CV
  02_Tuning_PatchTST.ipynb           # 45 kasus CV
  03_Tuning_Hierarchical.ipynb       # 45 kasus CV
  04_Final_dan_Evaluasi.ipynb        # seleksi sampai laporan
  05_Visualisasi_Proses_dan_Hasil.ipynb  # grafik tanpa training
  _manual_runner.py                # output langsung, log, interupsi
results/finetune_revision/<run_id>/
  protocol.json
  folds/
  augmentation/
  encoder/
    S01/fold_1/                   # config, metrics, history, best checkpoint
    cv_fold_results.csv           # 45 baris
    cv_summary.csv                # 9 baris
    best_config.json
    final/                        # model, scaler reference, label mapping
    test/                         # metrics dan patient predictions
  patchtst/
  hierarchical/
  selected_model.json
  comparison/
  visualisasi/                       # 19 grafik PNG untuk Bab IV
```

Runner sudah tersedia dan hasil tahap utama lengkap. Perintah di bawah adalah rujukan jika perlu melanjutkan atau memeriksa ulang kasus; menjalankannya tidak diperlukan untuk membaca hasil saat ini:

```bash
.venv/bin/python scripts/02_prepare_fold_augmentation.py --device mps --all-folds --resume
.venv/bin/python scripts/03_tune.py --model encoder --scenario S01 --fold 1 --resume
.venv/bin/python scripts/03_tune.py --model encoder --all-scenarios --all-folds --resume
.venv/bin/python scripts/03_tune.py --model patchtst --all-scenarios --all-folds --resume
.venv/bin/python scripts/03_tune.py --model hierarchical --all-scenarios --all-folds --resume
.venv/bin/python scripts/04_select_best.py --all-models
.venv/bin/python scripts/05_train_final.py --model encoder --seed 42 --resume
.venv/bin/python scripts/06_evaluate_test.py --model encoder --seed 42
```

Setiap runner membaca artefak tahap sebelumnya. Notebook per model mengatur eksekusi dan membaca hasil; definisi model atau fungsi training tidak diduplikasi di tiga notebook. Notebook 05 membaca hasil dan membuat gambar tanpa GPU. Artefak lama tetap dipertahankan, dan hasil revisi menggunakan lokasi berbeda agar bukti pengerjaan terdahulu tidak tertimpa.

## 7. Checklist pengerjaan berikutnya

- [x] Dataset, diagnosis, dan keluaran preprocessing tersedia.
- [x] Pemetaan tiga kelas dan split pasien tersedia.
- [x] Tiga arsitektur telah diimplementasikan dan baseline dijalankan.
- [x] GAN, CV tuning terbatas, retraining, inferensi, dan SHAP pernah dijalankan.
- [x] Tetapkan grid revisi: learning rate `[1e-4, 5e-4, 1e-3]` × batch size `[16, 32, 64]`, dengan sembilan skenario.
- [x] Bekukan protokol software dan siapkan draf perbaikan metodologi proposal.
- [x] Terapkan grid revisi pada kode training.
- [ ] Terapkan draf perubahan Tabel 3.3 dan metodologi ke source proposal editable.
- [x] Samakan definisi arsitektur baseline dan fine-tuning.
- [x] Pecah kode serta runner per tahap/per model.
- [x] Siapkan notebook manual dengan output epoch langsung, log, dan resume checkpoint.
- [x] Siapkan manifest pasien dan scaler kelima training fold.
- [x] Selesaikan training GAN kelima fold dan GAN final.
- [x] Jalankan 9 skenario × 5 fold pada setiap model (135/135 kasus).
- [x] Simpan hasil tiap fold dan ringkasan mean ± SD.
- [x] Tetapkan konfigurasi dan model utama dari CV.
- [x] Retrain final pada seluruh development dengan aturan epoch yang dibekukan.
- [x] Evaluasi hold-out tanpa k-fold untuk satu seed yang telah ditetapkan.
- [x] Perbarui pembanding baseline dan SHAP pada protokol yang konsisten.
- [x] Buat tabel hasil dan 19 grafik visualisasi dari seluruh proses.
- [x] Audit numerik kehalusan GAN pada lima fold/final dan konsistensi sadapan ekstremitas.
- [ ] Evaluasi morfologi dan penyebab selisih GAN bersama ahli ECG sebelum mengklaim mutu sintetis.
- [ ] Rancang ablation baru yang memisahkan efek GAN, penanganan kelas, dan konfigurasi classifier.
- [ ] Terapkan draf revisi Bab III pada source naskah yang dapat diedit.
- [ ] Tulis pembahasan Bab IV/V dan keterbatasan dari hasil revisi aktual.

Prioritas berikutnya: **tindak lanjuti [daftar perbaikan](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/PRIORITAS_PERBAIKAN_DAN_EKSPERIMEN.md>) → terapkan metodologi Bab III → tulis hasil Bab IV dan kesimpulan/keterbatasan Bab V**. Jika pembimbing meminta pengulangan seed atau pasien eksternal baru, tetapkan protokol tambahan sebelum menggunakannya untuk penilaian.

## 8. Sumber

- [Proposal TA](</Users/daniandraaa/Desktop/TA GUWEH/docs/proposal/PROPOSAL TA.pdf>), terutama bagian 1.4-1.5, Gambar 3.3, dan Tabel 3.3; nomor halaman dijelaskan pada bagian 1.
- Kode, keluaran notebook, metadata split, dan CSV hasil project yang dirujuk di atas. Hasil diperiksa sebagai artefak tersimpan, tidak direproduksi pada audit ini.
- Dokumen `docs/referensi/Validasi Model Berbasis Pasien.pdf` merupakan salinan percakapan dan digunakan sebagai konteks tambahan, bukan sebagai sumber penelitian ilmiah.
- [Dokumentasi resmi scikit-learn tentang cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) menjelaskan pemilihan parameter pada validation/CV dan peran test terpisah.
- [Dokumentasi resmi scikit-learn tentang data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) menjelaskan bahwa transformasi yang belajar dari data harus di-fit pada data training. Penerapan prinsip ini pada GAN per fold merupakan rekomendasi metodologis untuk kode project ini.
