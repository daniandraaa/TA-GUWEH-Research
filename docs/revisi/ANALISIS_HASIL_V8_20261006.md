# Analisis hasil classifier V8 — 6 Oktober 2026

**Notebook 41–45 selesai tanpa error tersimpan; seluruh 30 selector inner dan 90 pembanding outer valid. GAN belum memberi manfaat yang konsisten terhadap kedua kontrol dan kedua kelompok RVOT.** Tidak perlu mengulang training V8.

Run: `20261005_paired_classifier_augmentation_v8`. Fingerprint: `b5b4b959b108915f3650c2e24edade19054ff026770c6a77af2fffa9014f1524`.

## Pekerjaan yang sudah selesai

| Notebook | Sel kode | Hasil |
|---|---:|---|
| 41 | 4/4 | Prepare dan smoke MPS kedua model × tiga arm lulus, batch 8 × 4 × 12 × 5000 |
| 42 | 5/5 | Pilot dua model, dua selector + enam pembanding dan laporan selesai |
| 43 | 3/3 | Hierarchical: 15 selector + 45 pembanding selesai |
| 44 | 3/3 | Encoder: 15 selector + 45 pembanding selesai |
| 45 | 4/4 | Seluruh hasil, OOF, metrik, recall dan pooling selesai diaudit |

Total **120 fit, 2742 epoch dan 59.175 update optimizer** di luar enam update smoke disposable. Bobot terpilih seluruh 120 case finite; 90 bobot outer identik checkpoint epoch terakhir. Ada **4140 baris prediksi OOF**: 230 pasien × dua model × tiga seed × tiga arm. Pasien yang sama berulang; ini bukan 4140 pasien independen. Dukungan kelas development: LVOT 57, RVOT cusp 103, RVOT septal/freewall/other 70.

## Hasil utama: mean macro-F1 fold/seed

Setiap baris merangkum lima fold × tiga seed. SD deskriptif, bukan standard error dari 15 sampel independen.

| Model | Arm | Mean macro-F1 ± SD | Recall cusp | Recall septal/freewall/other |
|---|---|---:|---:|---:|
| Encoder | Real-only | 0,5170 ± 0,0887 | 41,37% | 46,19% |
| Encoder | Augmentasi real | 0,5292 ± 0,0575 | 48,38% | 44,76% |
| Encoder | GAN | 0,5316 ± 0,0659 | 59,56% | 38,57% |
| Hierarchical | Real-only | 0,5081 ± 0,0958 | 42,54% | 49,05% |
| Hierarchical | Augmentasi real | 0,5339 ± 0,1133 | 54,25% | 42,86% |
| Hierarchical | GAN | 0,4796 ± 0,0718 | 52,97% | 28,57% |

**Encoder:** GAN naik 0,0146 dibanding real-only, tetapi hanya 0,0024 dibanding augmentasi real. Mean fold-F1 versus real-only membaik hanya pada seed 43 (+0,0664); seed 42/44 turun −0,0116/−0,0112. Recall cusp naik **18,19 poin persentase**, sementara recall other turun **7,62 poin**. Gain agregat belum cukup untuk memilih GAN sebagai perbaikan stabil.

**Hierarchical:** GAN turun 0,0285 dibanding real-only dan 0,0543 dibanding augmentasi real. Mean fold-F1 turun pada seluruh tiga seed; recall other turun **20,48 poin persentase**. Recipe GAN V8 tidak diprioritaskan sebagai arm final Hierarchical.

Prediksi bergeser ke cusp. Pada 210 evaluasi berulang untuk 70 pasien other × tiga seed, jumlah salah diprediksi cusp meningkat **Encoder 99 → 120**, **Hierarchical 81 → 133** pada real-only → GAN. Ini pola kesalahan deskriptif, bukan estimasi dari 210 pasien berbeda.

## Pooled OOF memberi perspektif tambahan

Pooled OOF menghitung macro-F1 dari seluruh 230 pasien setelah lima fold digabung **per model/seed/arm**, kemudian merangkum tiga seed. Ini berbeda dari mean lima F1 fold karena F1 nonlinear; jangan mengganti definisi metrik agar hasil terlihat lebih tinggi.

| Model | Real-only | Augmentasi real | GAN |
|---|---:|---:|---:|
| Encoder, mean pooled OOF | 0,5270 | 0,5473 | 0,5703 |
| Hierarchical, mean pooled OOF | 0,5412 | 0,5613 | 0,5056 |

Pooled OOF Encoder GAN lebih tinggi dari real-only pada semua tiga seed, dan dari augmentasi real pada dua seed. Ini sinyal yang layak dicatat, bersama hasil mean fold dan penurunan recall other. Hierarchical tetap negatif. Belum cukup untuk menyatakan manfaat lokalisasi yang konsisten atau validasi klinis sintetis.

## Mengapa sekitar 0,5 berbeda dari sekitar 0,65 pada V4?

Audit membandingkan real-only V8 dengan kontrol K4 V4 pada model, LR, pooling, seed, scaler dan pasien training yang sama. **Loss dan akurasi training cocok persis pada semua epoch yang sama-sama tersedia di 30 lintasan.** Untuk **24 kasus** yang mempunyai epoch pilihan V8 dalam history V4, **F1 pada epoch tersebut juga identik**. Enam kasus lainnya melatih V8 melewati panjang history V4 yang dulu early-stop.

V4 memilih checkpoint dengan F1 terbaik pada **outer validation**, lalu melaporkan F1 pada kelompok yang sama. V8 memilih epoch memakai **37 pasien inner**, kemudian refit dari awal pada 183–185 outer-training dan menilai outer setelah fixed budget selesai. Karena pemilihan checkpoint berbeda, perbandingan 0,65 V4 → sekitar 0,51 V8 tidak mengisolasi efek GAN. Baseline V8 tanpa GAN juga turun. Audit ini tidak menemukan kerusakan kode training, scaler atau crop real-only.

Mean kontrol V4 lima fold/tiga seed: Encoder 0,6534, Hierarchical 0,6525; ini hasil dengan aturan pemilihan outer lama. Angka V8 tidak boleh “dikoreksi” dengan kembali memilih epoch terbaik dari outer. Pemisahan epoch V8 lebih ketat, meski seluruh estimasi tetap eksploratif karena pemilihan LR/pooling V4 dan desain sebelumnya telah menggunakan development.

Epoch V8 bervariasi: **Encoder 2–65, median 14; Hierarchical 3–49, median 16**. Tidak ada selector memilih epoch 1. Mean inner F1 Encoder 0,6594, Hierarchical 0,6369; berbeda dari skor refit outer. Ukuran inner yang kecil, heterogenitas pasien dan transfer pilihan epoch dari 146–148 ke 183–185 training adalah **hipotesis stabilitas yang perlu diuji**, bukan penyebab pasti atau alasan menambah epoch tanpa kontrol.

## Kontrol dan integritas yang diperiksa

- Rantai signature sumber V7/V8, case, scaler, checkpoint, prediksi dan manifest valid. Seluruh tabel utama asli direkonstruksi dan cocok.
- Epoch, LR schedule, jumlah update, exposure pasien anchor, jumlah window dan eligible bag identik antarketiga arm pada setiap pasangan model/fold/seed.
- Seluruh pasien anchor muncul sekali per epoch. Donor real hanya GAN-fit, kondisi donor/sintetis sesuai kelas, dosis dan indeks bank tercatat.
- Penggantian satu window dilakukan hanya pada bag K4 penuh. Fraksi penggantian semua window aktual **23,25–23,91%**, lebih rendah dari 25% karena bag pendek tidak diganti; pasien pendek tetap dipakai.
- Prediksi outer menggunakan seluruh window pasien real dengan pooling global. Mean pooling Hierarchical, bobot sum-to-one, label, enam metrik dan recall cocok. Tidak memilih checkpoint dari outer dan tidak membaca test.
- **350 file historis dan 1259 file kode/notebook/hasil V8 asli** yang dipantau tetap utuh. Audit baru memakai checkpoint CPU dan prediksi tersimpan, tanpa training atau inference waveform baru.

[Audit mesin dan hash sumber](../../results/research_v8/20261005_paired_classifier_augmentation_v8/analysis_20261006_complete/audit.json) · [laporan dan grafik](../../results/research_v8/20261005_paired_classifier_augmentation_v8/analysis_20261006_complete/REPORT.md) · [efek per seed](../../results/research_v8/20261005_paired_classifier_augmentation_v8/analysis_20261006_complete/paired_seed_effects.csv) · [perbandingan epoch V4/V8](../../results/research_v8/20261005_paired_classifier_augmentation_v8/analysis_20261006_complete/epoch_selection_v4_v8.csv).

## Keputusan dan fase berikutnya

Bekukan V8 sebagai ablation lengkap, termasuk hasil negatif. **Tidak perlu mengulang 41–45, menambah rasio GAN, memilih seed terbaik atau menjalankan final/test sekarang.** GAN V7 tetap hasil perbaikan mutu komputasional; mutu tersebut tidak otomatis membantu classifier. Angka V8 belum membuktikan batas maksimal performa.

Prioritas berikut adalah **stabilitas pemilihan epoch dan generalisasi real-only → representasi/pretraining yang sesuai sumber → baru ablation augmentasi lanjutan bila ada dasar**. Penggunaan bobot awal saat ini masih from scratch; eksperimen fine-tuning pretrained membutuhkan sumber dan pemisahan data yang benar-benar didefinisikan. Perlu riset/recipe baru sebelum training berikut, bukan sekadar memperbanyak sintetis pada recipe V8.

[Urutan fase setelah V8](FASE_SETELAH_V8_20261006.md). Tidak tersedia notebook training 46 pada audit ini. Hasil model final/test historis tidak berubah.
