# Analisis hasil GAN V7 — 5 Oktober 2026

**Notebook 37–40 selesai tanpa error tersimpan, dan kelima fold lolos screening mutu yang dihitung ulang. Fase GAN/QC selesai; belum ada skor classifier baru.**

Run: `20261004_fit_distribution_gan_v7`. Fingerprint: `6e6a9308d65a3160a17aab54e09ea6149d947ceca72632001a0e62f6bfd11125`.

## Pekerjaan yang sudah selesai

| Notebook | Sel kode dieksekusi | Hasil |
|---|---:|---|
| 37 | 4/4 | Persiapan dan smoke MPS lulus; gradien objective CPU tetap mencapai generator MPS |
| 38 | 5/5 | Pilot fold 1, 100 epoch, sintesis dan QC selesai/PASS |
| 39 | 4/4 | Grafik serta perbandingan V6/V7 tersedia dan cocok dengan tabel sumber |
| 40 | 7/7 | Fold 2–5, sintesis, QC, laporan dan status selesai/PASS |

| Fold | Pasien fit/ref/outer | Epoch | Update G dan D masing-masing | Sintetis | Keluarga gagal | Fraksi fitur gagal terbesar |
|---|---|---:|---:|---:|---:|---:|
| 1 | 147/37/46 | 100 | 3700 | 192 | 0/15 | 19,7% |
| 2 | 146/37/47 | 100 | 3700 | 192 | 0/15 | 10,6% |
| 3 | 147/37/46 | 100 | 3700 | 192 | 0/15 | 19,0% |
| 4 | 148/37/45 | 100 | 3700 | 192 | 0/15 | 8,3% |
| 5 | 147/37/46 | 100 | 3700 | 192 | 0/15 | 10,6% |

Total **500 epoch, 18.500 update G dan D masing-masing, 960 window sintetis**, yaitu 64 per kelas per fold. Sintetis tidak dihitung sebagai pasien baru. Jumlah `seconds` pada history sekitar 31,0 menit untuk epoch seluruh fold; belum termasuk prepare, checkpoint di luar penghitung epoch, generate, QC dan report.

## Perbaikan terukur terhadap V6

V6 hanya mempunyai GAN lengkap fold 1. Pada fold yang sama, dengan data, seed, arsitektur, anggaran dan ambang QC tetap, kegagalan kombinasi kelas–keluarga turun **9/15 → 0/15**. Mean jarak Wasserstein ternormalisasi membaik pada semua 12 pasangan kelas–keluarga non-limb. Ini perbandingan paket objective, belum ablation kontribusi tiap komponen.

Tabel berikut adalah **persentase fitur yang gagal dalam setiap keluarga**, bukan akurasi klasifikasi.

| Kelas | Amplitudo V6 → V7 | Roughness V6 → V7 | Korelasi V6 → V7 | Spektrum V6 → V7 |
|---|---:|---:|---:|---:|
| LVOT | 25,0 → 0,0% | 66,7 → 0,0% | 34,8 → 4,5% | 35,7 → 0,0% |
| RVOT cusp | 41,7 → 8,3% | 47,9 → 6,3% | 51,5 → 19,7% | 19,0 → 3,6% |
| RVOT septal/freewall/other | 45,8 → 0,0% | 33,3 → 0,0% | 45,5 → 15,2% | 22,6 → 2,4% |

Contoh median pada RVOT cusp, lead V2 fold 1: rasio derivative RMS terhadap signal RMS real 0,02278, V6 0,03549, V7 0,02509; korelasi V1–V2 real 0,9084, V6 0,3778, V7 0,8563; rentang p95−p05 real 3,9687, V6 2,9025, V7 3,9162 pada skala GAN global. Median relatif daya 40–100 Hz real 0,0424%, V6 1,4386%, V7 0,1242%: membaik tetapi belum identik. Unit amplitudo tidak diasumsikan mV.

## Apa yang diverifikasi

- Signature kode/protokol/runtime dan rantai preparation → checkpoint → sintesis → QC valid pada lima fold. Enam artefak split/skala/fitur real per fold identik V6.
- Setiap fold mempunyai epoch 1–100 berurutan, anggaran draw/update tepat dan loss finite. Exposure setiap epoch sesuai label dan hanya memuat pasien GAN-fit.
- Bobot generator/discriminator final finite dan identik `last.pt` epoch 100. Tidak memilih checkpoint dari validation classifier atau test.
- Fitur sintetis dihitung ulang dari waveform 12 × 5000 native, kemudian seluruh jarak, threshold dan keputusan keluarga direkonstruksi. Semuanya cocok dengan hasil pengguna.
- Numerik, lead konstan, duplikasi, exact-copy dan nearest-neighbour native terhadap seluruh window GAN-fit dihitung ulang. Semua lolos. Minimum relative RMSE ke training per fold 0,6854/0,7854/0,6823/0,8908/0,7870, di atas ambang near-copy 0,01.
- 163 file historis terlindungi dan 176 file kode/notebook/hasil V7 yang dicatat sebelum audit tetap utuh. Audit hanya menulis subfolder terpisah, tanpa training baru atau akses waveform test/outer-validation untuk fold yang diaudit.

[Audit mesin dan hash sumber](../../results/research_v7/20261004_fit_distribution_gan_v7/analysis_20261005_complete/audit.json) · [laporan/grafik seluruh fold](../../results/research_v7/20261004_fit_distribution_gan_v7/analysis_20261005_complete/REPORT.md) · [tabel ringkas](../../results/research_v7/20261004_fit_distribution_gan_v7/analysis_20261005_complete/fold_summary.csv).

## Kelemahan yang tetap perlu dilaporkan

PASS berarti maksimal 25% fitur gagal per kelas/keluarga. Masih terdapat **141 dari 3390 pemeriksaan fitur** yang gagal, meskipun 75/75 kombinasi kelas–keluarga memenuhi ambang. Fitur antar-lead/fold saling terkait; proporsi ini bukan estimasi akurasi klinis atau 3390 sampel independen. Korelasi RVOT cusp fold 1 dan spektrum RVOT septal/freewall/other fold 3 masih prioritas observasi.

Objective dan QC menilai keluarga fitur serupa; QC pasien referensi terpisah belum menilai seluruh morfologi ECG atau kebenaran diagnosis sintetis. Grafik yang telah diperiksa memakai contoh deterministik pertama, tidak diseleksi dari waveform terbaik. Tanpa ahli ECG, jangan mengklaim validasi PVC/anatomi klinis berdasarkan grafik atau PASS ini.

Pemeriksaan copy full-window tanpa phase alignment belum menilai setiap bentuk memorisasi atau membuktikan privasi. Satu seed GAN per fold belum mengukur variasi generator pada data fold yang sama. Recipe V7 dipilih setelah kegagalan V6, sehingga hasil merupakan eksperimen eksploratif pada development.

## Keputusan dan langkah selanjutnya

**Pertahankan GAN V7 dan jangan mengulang 37–40.** Perbaikan mutu cukup untuk melanjutkan eksperimen manfaat augmentasi secara komputasional, tanpa menganggap kinerja classifier pasti meningkat atau target 0,9 telah tercapai.

Fase berikut ialah protokol dan endpoint classifier pada paket/run terpisah: **K4 real-only → kontrol augmentasi real → real+GAN**, mula-mula Hierarchical dan Encoder pada lima fold/tiga seed, validation berupa pasien real. Pertahankan model, fold, seed, optimizer, pooling, scaler, anggaran update dan arti batch pasien yang sebanding; tetapkan porsi sintetis sebelum training, dan catat bag artificial tanpa menganggapnya pasien baru. Tinjau macro-F1 pasien, recall kedua RVOT, prediksi OOF, variasi seed dan biaya.

Endpoint classifier baru belum dibuat pada audit ini. Skor final/test belum berubah. Manfaat yang konsisten perlu dibuktikan sebelum ablation komponen dan seed GAN tambahan; test historis tidak digunakan memilih arm atau seed. [Rancangan fase berikut](FASE_CLASSIFIER_SETELAH_V7_20261005.md).
