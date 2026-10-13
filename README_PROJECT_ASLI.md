# Project TA — revisi fine-tuning

**V13 (Opsi A, sesuai proposal) berjalan sejak 7 Oktober 2026:** baseline (data asli, LR 1e-3, batch 32) vs fine-tuning (GAN V7 lolos QC + Grid Search 9 LR × batch) dengan resep identik, **3 model × 5 fold × 3 seed (450 fit CV)**, 30 model final (5 seed), test 59 pasien, uji Wilcoxon/Friedman/McNemar + bootstrap, dan SHAP enam model. Mulai dari [notebook 69](notebooks/69_V13_Persiapan_dan_Smoke.ipynb) → 70–72 (CV) → 73 (seleksi, GAN final, final) → 74 (test + Bab IV) → 75 (SHAP). [Panduan V13](docs/revisi/PANDUAN_MANUAL_V13.md).

**Notebook dirapikan — 7 Oktober 2026:** 52 notebook versi lama (V1, V3, V5–V6, V8–V12) dipindah ke [arsip/notebooks_versi_lama_20261008](arsip/notebooks_versi_lama_20261008/README.md). `notebooks/` kini hanya berisi V2 (06–10), V4 (20–27), V7 (37–40) dan V13 (69–75). Hasil di `results/` tidak diubah; tautan notebook lama di bawah mengarah ke folder arsip tersebut.

**V12 selesai dan diaudit — 7 Oktober 2026:** notebook **62–67 selesai**, **40/40 fit CV + 1 deployment**; 100 skor evaluasi/OOF/sample training terverifikasi. Mean fold F1 PVC **morfologi 0,6793; morfologi+RR 0,7897** (pooled OOF RR 0,8041). **MIT-BIH F1 0,3998, gate FAIL; 68 melewati ekspor**. Sebanyak 2.316 PVC terlewat detector dan 4.239 false alarm berasal dari kandidat unmatched. **Tidak mengulang 62–68 pada recipe yang sama**; prioritas berikut ialah benchmark/perbaikan detector dan generalisasi classifier pada run baru. [Analisis lengkap](docs/revisi/ANALISIS_HASIL_V12_20261007.md) · [panduan run yang selesai](docs/revisi/PANDUAN_MANUAL_V12.md). Ini skor jenis beat, belum origin final atau masukan lokal baru.

**Kajian literatur — 7 Oktober 2026:** [perbandingan paper dan rancangan fase berikut](docs/revisi/KAJIAN_LITERATUR_CNN_TRANSFORMER_PVC_20261007.md) mencakup CNN–Transformer, masukan PVC–sinus, fitur antarlead, detektor pendukung, evaluasi eksternal dan koreksi referensi Shahsavari menjadi jurnal IEEE 2024. Prioritas penelitian ialah identitas beat dan pemisahan kedua kelompok RVOT dengan ablation terkontrol; implementasi fase pertama tersedia pada V12 di atas. [BibTeX](docs/revisi/REFERENSI_PVC_CNN_TRANSFORMER_20261007.bib).

**V11 tiga arm fitur selesai — 7 Oktober 2026:** notebook **60 dan 61 masing-masing 11/11 sel tanpa error**; **150 fit (135 inner + 15 outer)** dan tiga OOF × 230 pasien terverifikasi. Mean fold macro-F1: **QRS/RR 0,4210; morfologi 0,5555; morfologi + RR 0,5532**. Tambahan RR belum meningkatkan mean; kesalahan terbesar antara kedua kelompok RVOT. **Tidak perlu mengulang tiga arm fitur.** Empat arm encoder/fusion masih tertunda karena cache V10 belum tersedia, dengan 200 fit terencana tersisa; belum ada model final/test baru. [Analisis lengkap dan langkah berikutnya](docs/revisi/ANALISIS_HASIL_FITUR_V11_20261007.md) · [panduan](docs/revisi/PANDUAN_MANUAL_V11.md). Catatan persiapan/perbaikan di bawah adalah riwayat sebelum hasil ini.

**Perbaikan notebook 58 — 7 Oktober 2026:** download LUDB 200 rekaman sudah lengkap; percobaan pertama berhenti karena disk, berikutnya karena cache encoder V10 tidak tersedia. Runner perbaikan memisahkan dependency LUDB/morfologi dari embedding dan menangani anotasi batas yang tidak lengkap tanpa menebak ground truth/mengubah threshold. **Restart Kernel → 57 → 58 dari sel pertama**, run baru berakhiran `_r1`; `DOWNLOAD_LUDB=False`. [Diagnosis/panduan](docs/revisi/PERBAIKAN_NOTEBOOK_58_CACHE_DAN_ANOTASI_20261007.md). Source/log lama tetap dipertahankan. Belum ada hasil benchmark waveform atau training V11 baru; encoder/fusion tetap membutuhkan pemulihan cache yang cocok dengan audit.

**Fase aktif V11 — 6 Oktober 2026:** **notebook 56 selesai**, enam fit replay cocok dan pemeriksaan encoder pada tiga pasien training lulus; gap **tiga refit outer fold1/seed42** sekitar **0,318–0,391 macro-F1**. Fitur morfologi/QRS/RR, validasi onset–offset LUDB, head L2 dan fusion cache encoder kini tersedia pada **notebook 57–61**. Mulai **57 → 58 → 59**, kemudian pilot **60** dan CV **61** bila pengukuran PASS. **350 fit CPU terencana termasuk 70 pilot; belum ada training ECG asli/hasil V11.** [Panduan manual V11](docs/revisi/PANDUAN_MANUAL_V11.md) · [prioritas dan riset fitur](docs/revisi/RISET_DAN_FASE_FITUR_MORFOLOGI_V11_20261006.md). Semua pasien development tetap dipertahankan; kandidat QRS belum terverifikasi PVC. Hasil lama tidak perlu diulang; final/historical test dinonaktifkan pada V11.

**V10 selesai dan diaudit — 6 Oktober 2026:** notebook **51–55** lengkap tanpa error; **99 selector inner + 31 refit outer**, 130 checkpoint/scaler, 2.945 epoch dan 33.832 update terverifikasi. Mean macro-F1 **random frozen 0,4415**, **pretrained frozen 0,3744**. Tail fold1/seed42 **0,3928**, turun dari pretrained frozen pada kasus sama **0,4454**. [Analisis V10](docs/revisi/ANALISIS_HASIL_V10_20261006.md). **Tidak mengulang 51–55**. Berikutnya [notebook 56](notebooks/56_Diagnosis_Transfer_V10_Tanpa_Training_Ulang.ipynb): diagnosis checkpoint/gap training–validasi/CPU–MPS/cache, **inference manual tanpa training ulang atau test**. [Panduan diagnosis](docs/revisi/PANDUAN_DIAGNOSIS_V10_20261006.md). Independensi pasien pretraining belum terverifikasi; belum ada hasil final/test baru.

**V9 selesai dan diaudit — 6 Oktober 2026:** notebook **46–50** lengkap, **20 fit baru + 10 referensi V8**, 30 checkpoint dan seluruh prediksi terverifikasi. Mean fold/seed macro-F1 Encoder **0,5170 → 0,5311**, Hierarchical **0,5081 → 0,4943**; recall RVOT other turun **9,52 / 7,14 poin persentase**. Median belum menjadi perbaikan final yang konsisten. [Analisis V9](docs/revisi/ANALISIS_HASIL_V9_20261006.md). **Tidak mengulang 46–50**; berikutnya audit representasi pretrained dan protokol fine-tuning terkontrol. Belum ada model final/test baru.

**Riwayat persiapan V9:** notebook 46–50 menguji median tiga epoch inner dengan 10 referensi V8 dan 20 fit baru. Seluruhnya kini selesai dan diaudit pada pembaruan di atas. [Panduan V9](docs/revisi/PANDUAN_MANUAL_V9.md) dipertahankan sebagai rujukan historis; pekerjaan aktif beralih ke V10.

**V8 selesai dan diaudit — 6 Oktober 2026:** notebook **41–45** lengkap; **30 selector inner + 90 pembanding** serta prediksi/checkpoint terverifikasi. Mean fold/seed macro-F1: Encoder real-only **0,5170**, real augmentation **0,5292**, GAN **0,5316**; Hierarchical **0,5081 / 0,5339 / 0,4796**. GAN belum konsisten dan recall RVOT other turun pada keduanya. [Analisis V8](docs/revisi/ANALISIS_HASIL_V8_20261006.md) · [fase berikut](docs/revisi/FASE_SETELAH_V8_20261006.md). **Tidak perlu mengulang 41–45; belum ada final/test baru.**

**Riwayat implementasi V8 — 5 Oktober 2026:** classifier K4 real-only/kontrol augmentasi real/real+GAN sudah disiapkan pada paket/run baru. Mulai **41 → 42** untuk smoke dan pilot, lalu **43 → 44 → 45** untuk full comparison dan analisis. Epoch dipilih dari inner real-only dan dipakai sama antarketiga arm. Anggaran **30 selector + 90 pembanding**, termasuk pilot; **training data asli tetap manual**. [Panduan V8](docs/revisi/PANDUAN_MANUAL_V8.md). GAN V7 lengkap dan tidak perlu diulang.

**V7 selesai dan diaudit — 5 Oktober 2026:** notebook **37–40** lengkap tanpa error tersimpan; **5/5 GAN, 500 epoch dan 960 window sintetis** terverifikasi. QC dihitung ulang dari waveform native: seluruh fold PASS; kegagalan kelas–keluarga fold 1 membaik **V6 9/15 → V7 0/15**, dengan ambang tetap. [Analisis V7](docs/revisi/ANALISIS_HASIL_V7_20261005.md) · [fase classifier berikut](docs/revisi/FASE_CLASSIFIER_SETELAH_V7_20261005.md). **Tidak perlu mengulang 37–40.** Belum ada skor classifier/final/test baru; endpoint augmentasi classifier kini tersedia pada V8/notebook 41–45.

**Riwayat V6:** notebook 33–35 selesai; GAN fold 1 selesai 100 epoch tetapi screening mutu FAIL. Notebook 36 berhenti sesuai gate sebelum fold 2–5. [Diagnosis V6](docs/revisi/ANALISIS_PILOT_GAN_V6_20261004.md). Hasil V6 dipertahankan sebagai pembanding; tidak perlu mengulang 34/36 pada recipe yang sama.

> **Status V4 — 3 Oktober 2026:** notebook **20–27 selesai**; diagnosis 45/45, CV 450/450, stability 75/75, final 25/25 dan test 25/25. Audit prediksi/checkpoint lulus. Model primer Hierarchical: CV macro-F1 0,6616; test seed 42 0,6198; mean lima seed 0,5421 ± 0,0845. [Analisis hasil dan prioritas selanjutnya](docs/revisi/ANALISIS_HASIL_V4_20261003.md). [Panduan manual](docs/revisi/PANDUAN_MANUAL_V4.md) menjadi rujukan alur yang sudah selesai. Target 0,9 belum tercapai.

**V5 selesai dan diaudit — 4 Oktober 2026:** notebook **29–32**, kedua smoke MPS, **30/30 training baru** dan 30 kontrol terverifikasi. K8 menurunkan mean development macro-F1 Hierarchical **0,6525 → 0,6367**, Encoder **0,6534 → 0,6427**; **pertahankan K4**. [Analisis lengkap V5](docs/revisi/ANALISIS_HASIL_V5_20261004.md). Tidak ada final/test baru. Notebook 28 selesai **230/230 development dan 12 pratinjau**; [dasar audit](docs/revisi/ANALISIS_AUDIT_INPUT_V4_20261003.md).

> **V3 yang sudah selesai:** **630/630 CV, 20/20 final, 20/20 test**, analisis 18/SHAP 19 dan benchmark INCART 17 terverifikasi. Model primer Hierarchical seed 42 test macro-F1 0,5837. INCART F1 0,96862/recall V 92,79% adalah metrik deteksi beat. Hasil lama dipertahankan; notebook 12–19 tidak perlu diulang. [Analisis V3](docs/revisi/ANALISIS_HASIL_V3_20261002.md) · [INCART](docs/revisi/ANALISIS_BENCHMARK_INCART_20261002.md).

> **Analisis hasil V2 tersedia:** 270 CV dan seluruh final/test sudah lengkap. Buka [Notebook 10](notebooks/10_Analisis_Hasil_V2.ipynb) atau [laporan 1 Oktober 2026](docs/revisi/ANALISIS_HASIL_V2_20261001.md) untuk tabel, diagnosis per kelas, variasi seed, dan prioritas lanjutan.

**Notebook 20–32 sudah selesai; tidak perlu training ulang.** Pilot GAN/QC kini tersedia pada V6, kemudian ablation classifier dengan baseline K4 setelah mutu sintetis diperiksa. Evaluasi diperkuat dan Bab III–V masih pekerjaan lanjutan; [daftar fase](docs/revisi/FASE_SETELAH_V4_20261003.md). [Panduan V5](docs/revisi/PANDUAN_MANUAL_V5.md) menjadi referensi alur yang sudah selesai.


> **Riwayat V2 yang sudah selesai:** [Notebook 06 — Audit sumber dan preprocessing](notebooks/06_Audit_Sumber_dan_Preprocessing_V2.ipynb), kemudian 07–09. Sumber menegaskan 2.000 Hz; tiga kelas dikoreksi namanya sesuai anatomi dengan anggota tetap. [Panduan V2 dan batas interpretasi](docs/revisi/PANDUAN_REVISI_V2.md).


**Riwayat notebook 00–05:** untuk membaca visualisasi V1, buka [05_Visualisasi_Proses_dan_Hasil.ipynb](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>). Pilih Python `.venv` project dan jalankan sel satu per satu dengan **Shift+Enter**. Notebook 05 membaca hasil yang tersimpan tanpa training ulang.

## Riwayat urutan V1 (00–05)

| Urutan | Notebook | Pekerjaan | Syarat selesai |
|---|---|---|---|
| 00 | [Mulai di sini](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/00_Mulai_di_Sini.ipynb>) | Persiapan data, fold, scaler, dan GAN bersama | GAN kelima fold lengkap |
| 01 | [Tuning Encoder](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/01_Tuning_Encoder.ipynb>) | 9 skenario × 5 fold CNN-Transformer Encoder | 45/45 training CV |
| 02 | [Tuning PatchTST](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/02_Tuning_PatchTST.ipynb>) | 9 skenario × 5 fold CNN-PatchTST | 45/45 training CV |
| 03 | [Tuning Hierarchical](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/03_Tuning_Hierarchical.ipynb>) | 9 skenario × 5 fold CNN-Hierarchical Transformer | 45/45 training CV |
| 04 | [Final dan evaluasi](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/04_Final_dan_Evaluasi.ipynb>) | Seleksi, baseline pembanding, retraining, inferensi, SHAP, laporan | Seluruh hasil final tersedia |
| 05 | [Visualisasi proses dan hasil](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>) | Data, GAN, CV, final, test, SHAP | Grafik dan ringkasan untuk Bab IV |

Grid aktif: learning rate `[1e-4, 5e-4, 1e-3]` × batch size `[16, 32, 64]`. Model final dipilih dari CV development; test 20% dievaluasi tanpa k-fold.

## Posisi pengerjaan saat audit 30 September 2026

GAN lima fold dan final selesai; fine-tuning **135/135 CV**, baseline **15/15 CV**, enam model final, enam evaluasi test, serta SHAP dua model utama tersimpan. Hasil revisi diaudit pada [analisis eksperimen](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/ANALISIS_HASIL_EKSPERIMEN.md>). Model utama dari CV adalah Encoder; PatchTST mencatat F1 test tertinggi untuk pelaporan. Sembilan belas grafik PNG tersedia di folder `results/finetune_revision/20260929_grid9_v1/visualisasi/`, termasuk audit GAN seluruh fold dan konsistensi sadapan.

Jika perlu menelusuri V1, [notebook 00](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/00_Mulai_di_Sini.ipynb>) adalah titik mulai dan hasil yang lengkap akan dilewati. Tidak diperlukan training ulang untuk melihat visualisasi V1. Pekerjaan aktif V3 tercantum di bagian awal README ini.

## Letak file

| Lokasi | Isi |
|---|---|
| `notebooks/` | V1 00–05, V2 06–10, V3/SHAP/benchmark 11–17, analisis final V3 18 dan SHAP primer V3 19 |
| `research_v3/` | Paket training V3 yang dibekukan; jangan ubah selama melanjutkan run |
| `analysis_v3/` | Audit hasil tersimpan dan visualisasi tanpa training |
| `configs/` | Kandidat parameter dan aturan eksperimen |
| `src/` | Modul data, GAN, model, training, metrik, dan laporan |
| `scripts/` | Runner jika memilih menjalankan melalui Terminal |
| `docs/` | Proposal, referensi, fase revisi, dan draf metodologi |
| `arsip/` | Notebook historis dan skrip percobaan beserta keluaran lama |
| `results/finetune_revision/20260929_grid9_v1/` | Checkpoint, progres, log, dan hasil revisi |
| `results/research_v3/20261001_patient_r_candidates_v3/` | Seluruh CV/final/test V3, log, snapshot pilot dan analisis lengkap 2 Oktober |
| `results/detector_benchmark/incart/848d6db41c5d/` | 75 hasil benchmark, summary dan `analysis_20261002/` untuk audit, CI sumber Holter serta grafik kegagalan |
| `baseline_results/`, `finetune_results/` | Hasil eksperimen historis |
| `preprocessed_data/`, `PVCVTECGData/`, `PVCVTRawECGData/`, `Diagnosis.xlsx` | Data dan metadata yang digunakan project |

[Panduan menjalankan lengkap](</Users/daniandraaa/Desktop/TA GUWEH/README_REVISI.md>) · [Peta file](</Users/daniandraaa/Desktop/TA GUWEH/docs/PETA_FILE.md>) · [Fase revisi](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/FASE_REVISI_FINE_TUNING.md>) · [Audit skor 0,9](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/AUDIT_SKOR_09_DAN_ARAH_PUBLIKASI.md>) · [Prioritas perbaikan](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/PRIORITAS_PERBAIKAN_DAN_EKSPERIMEN.md>)
