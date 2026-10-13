# Menjalankan V9 — aturan epoch real-only

**V9 sudah selesai dan diaudit — 6 Oktober 2026:** notebook **46–50**, 20 fit baru dan 10 referensi terverifikasi. **Tidak perlu mengulang training.** Encoder naik sedikit tetapi recall other turun; Hierarchical turun. [Analisis hasil dan langkah berikut](ANALISIS_HASIL_V9_20261006.md). Petunjuk berikut dipertahankan sebagai riwayat menjalankan fase yang sudah selesai.

**Mulai dari notebook 46, lalu 47 → 48 → 49 → 50.** Gunakan kernel `.venv` project dan jalankan sel satu per satu dengan Shift+Enter. Training, smoke MPS dan inferensi outer sesudah training Anda jalankan manual.

V8 sudah selesai; tidak perlu mengulang notebook 41–45. V9 membandingkan aturan epoch V8 per seed dengan median tiga epoch inner per model/fold. Dua model × lima fold × tiga seed = **30 kasus pembanding**, terdiri dari **10 hasil V8 identik yang direferensikan + 20 training baru**. Tidak memakai skor outer untuk menentukan median. Ini menguji hipotesis kestabilan terhadap inisialisasi; tiga selector memakai split pasien inner yang sama.

| Notebook | Pekerjaan | Yang perlu dijalankan manual |
|---|---|---|
| [46 — diagnosis/persiapan](../../notebooks/46_Diagnosis_Epoch_dan_Persiapan_Real_Only_V9.ipynb) | Verifikasi sumber V8; salin scaler; rencana epoch dan lintasan inner | Sel smoke: satu update disposable per model; kemudian status |
| [47 — pilot](../../notebooks/47_Pilot_Fold1_Real_Only_V9.ipynb) | Fold 1 seed 42, kedua model | Hierarchical 18 epoch dan Encoder 10 epoch; audit pilot |
| [48 — Hierarchical](../../notebooks/48_Real_Only_Hierarchical_Lima_Fold_V9.ipynb) | Semua fold/seed Hierarchical | Sembilan fit baru tersisa setelah pilot; hasil lengkap dilewati |
| [49 — Encoder](../../notebooks/49_Real_Only_Encoder_Lima_Fold_V9.ipynb) | Semua fold/seed Encoder | Sembilan fit baru tersisa setelah pilot |
| [50 — analisis](../../notebooks/50_Analisis_Stabilitas_Epoch_Real_Only_V9.ipynb) | Rekonstruksi metrik, audit checkpoint dan visualisasi | Setelah 30/30 lengkap; tanpa training tambahan |

Persiapan dan diagnosis boleh dijalankan ulang karena cached/verifikasi. Status sesudah persiapan **10/30 pembanding, 0/20 fit baru**. Sesudah pilot **12/30, 2/20**; sesudah Hierarchical **21/30, 11/20**; sesudah Encoder **30/30, 20/20**. Smoke tidak dihitung sebagai training model penelitian.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | LR / pooling |
|---|---:|---:|---:|---:|---:|---|
| Hierarchical | 18 | 13 | 21 | 4 | 30 | 5e-4 / mean |
| Encoder | 10 | 11 | 10 | 19 | 18 | 1e-4 / attention |

Semua seed pada model/fold memakai epoch yang sama. Refit tetap dari inisialisasi baru dengan seluruh pasien training fold; bobot selector inner tidak digunakan. K4, batch delapan pasien, AdamW, class weight pasien, crop, scaler, seed dan cosine **T_max=100** tetap persis seperti V8. Mengganti T_max menjadi median epoch akan mengubah lintasan LR dan membatalkan ablation satu faktor ini. Paket V8 dipakai sebagai implementasi yang dibekukan; pada arm real-only tidak ada window sintetis yang dimasukkan.

Jangan menjalankan dua notebook training bersamaan. Jika Interrupt, tunggu proses anak berhenti; ulang sel yang sama untuk melanjutkan checkpoint epoch terakhir lengkap. Gunakan device `mps` yang sama dengan kontrol V8. Jangan mengubah LR, batch, kode, runtime atau run-id ketika melanjutkan run yang sama. Jika error, log menampilkan ekor traceback; periksa penyebab di atas pesan endpoint gagal.

Disk diperiksa sebelum persiapan dan setiap fit. Saat penyiapan, disk tersedia sekitar **4 GiB**, estimasi tambahan seluruh checkpoint sekitar **0,7 GiB**, dengan reserve **3 GiB + margin puncak 0,25 GiB**. Jika gate disk berhenti, pindahkan arsip yang sudah diverifikasi secara manual dan ulang sel. Jangan menghapus checkpoint/source V4–V8 untuk melewati gate; fase ini tidak mengunduh bobot foundation model atau menduplikasi cache waveform.

Run baru: `results/research_v9/20261006_inner_epoch_median_real_only_v9/`. Rencana: `epoch_plan.csv`; diagnosis: `diagnosis/REPORT.md`; hasil: `analysis_complete/REPORT.md`. Sepuluh referensi tetap bergantung pada checkpoint V8 asalnya.

Ukuran primer tetap **mean fold/seed macro-F1**, disertai kedua recall RVOT, variasi seed, collapse dan pooled OOF terpisah. Jangan memilih seed/ukuran metrik yang terlihat paling tinggi. Sepuluh referensi bukan replikasi baru; fold/seed tidak independen. V9 masih ablation development eksploratif setelah pemilihan LR/pooling V4, bukan estimasi nested sepenuhnya independen.

Sesudah notebook 50, lakukan [fase representasi/fine-tuning](RISET_REPRESENTASI_DAN_FINE_TUNING_20261006.md). Median dapat membantu, tidak berpengaruh atau menurunkan hasil; hasil negatif tetap informatif. Belum ada model final/test baru dan belum ada jaminan 0,9/Q1.
