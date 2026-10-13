# Peta file project TA

**V11 tiga arm fitur selesai — 7 Oktober 2026:** source/protokol `research_v11/` dan `research_v11_recovery/`, runner serta notebook hasil 57–61 tetap dipertahankan. Audit baru [analysis_v11/completed_features.py](../analysis_v11/completed_features.py) memeriksa 150 fit, 15 pilihan C dan tiga OOF × 230 pasien dari JSON/CSV/hash tanpa membuka model atau menjalankan training/inference waveform. Audit/CSV/CI/grafik ada di `results/research_v11/20261007_morphology_regularized_features_v11_r1/analysis_20261007_complete_features/`, terpisah dari analisis notebook. [Laporan lengkap dan langkah berikutnya](revisi/ANALISIS_HASIL_FITUR_V11_20261007.md). Empat arm encoder/fusion masih tertunda karena cache V10 belum tersedia.

**V9 selesai dan diaudit — 6 Oktober 2026:** paket `research_v9/`, runner dan notebook **46–50** dibekukan. `analysis_v9/audit_completed.py` memverifikasi 30 checkpoint/prediksi/OOF dan kesamaan lintasan epoch V8; keluaran terpisah di `results/research_v9/20261006_inner_epoch_median_real_only_v9/analysis_20261006_complete/`. [Analisis lengkap](revisi/ANALISIS_HASIL_V9_20261006.md). Tidak mengubah hasil asli atau menjalankan training/inference waveform baru.

**Tambahan V9 — 6 Oktober 2026:** `research_v9/`, `_v9_runner.py`, notebook **46–50** dan run `results/research_v9/20261006_inner_epoch_median_real_only_v9/`. Persiapan mereferensikan 10 kasus V8 yang identik; 20 training real-only baru tetap manual. Diagnosis epoch tersedia; analisis berpasangan/OOF sesudah lengkap. [Panduan](revisi/PANDUAN_MANUAL_V9.md) · [kajian pretrained](revisi/RISET_REPRESENTASI_DAN_FINE_TUNING_20261006.md). Paket/runtime/notebook hasil V2–V8 tetap dibekukan.

**V8 selesai dan diaudit — 6 Oktober 2026:** `research_v8/`, runner dan notebook **41–45** dibekukan setelah 120 fit. `analysis_v8/audit_completed.py` memverifikasi semua case/checkpoint/prediksi/OOF, anggaran berpasangan dan perbandingan epoch V4; audit/CSV/grafik terpisah di `results/research_v8/20261005_paired_classifier_augmentation_v8/analysis_20261006_complete/`. [Analisis](revisi/ANALISIS_HASIL_V8_20261006.md) · [fase berikut](revisi/FASE_SETELAH_V8_20261006.md). Tidak mengubah hasil asli atau menjalankan training/inference waveform baru.

**Tambahan V8 — 5 Oktober 2026:** `research_v8/`, `_v8_runner.py` dan notebook **41–45** menyediakan prepare/scaler, smoke, inner epoch selection, tiga arm classifier, resume serta audit OOF. Run/output baru: `results/research_v8/20261005_paired_classifier_augmentation_v8/`. [Panduan](revisi/PANDUAN_MANUAL_V8.md). Tidak mengubah source/checkpoint/notebook yang dijalankan pada V4–V7; belum ada skor classifier V8.

**V7 selesai dan diaudit — 5 Oktober 2026:** `research_v7/`, `_v7_runner.py` dan notebook **37–40** dibekukan setelah lima GAN/QC selesai/PASS. `analysis_v7/audit_completed.py` menghitung ulang QC native, memverifikasi checkpoint/exposure/hash dan menulis audit/CSV/heatmap terpisah di `results/research_v7/20261004_fit_distribution_gan_v7/analysis_20261005_complete/`. [Analisis](revisi/ANALISIS_HASIL_V7_20261005.md) · [fase classifier berikut](revisi/FASE_CLASSIFIER_SETELAH_V7_20261005.md). Endpoint classifier augmentasi tersedia pada V8; notebook training yang sudah dijalankan tidak diubah.

**Tambahan V6 — 4 Oktober 2026:** `research_v6/` memuat protokol, split fit/reference, GAN, training/resume, QC dan visualisasi. Runner `notebooks/_v6_runner.py`; notebook **33–35** untuk pilot, **36** untuk GAN fold2–5 sesudah PASS. Output terpisah di `results/research_v6/20261004_conditional_gan_qc_v6/`. [Panduan](revisi/PANDUAN_MANUAL_V6.md) · [kajian/fase](revisi/RISET_DAN_FASE_GAN_V6_20261004.md). Semua paket/notebook hasil V4/V5 tetap dipertahankan. Endpoint classifier GAN belum tersedia.

**V4 selesai — 3 Oktober 2026:** `research_v4/` dan `research_v4_mps_compat/` dipertahankan sesuai fingerprint. Notebook 20–27 sudah selesai; hasil asli di `results/research_v4/20261002_patient_mil_hybrids_v4/`. `analysis_v4/audit_completed.py` menghitung ulang metrik, memeriksa checkpoint/seleksi dan membuat analisis terpisah di `analysis_20261003_complete/` pada run tersebut. [Analisis hasil dan tindak lanjut](revisi/ANALISIS_HASIL_V4_20261003.md) · [panduan](revisi/PANDUAN_MANUAL_V4.md) · [kajian metode](revisi/RISET_METODE_DAN_PROTOKOL_V4_20261002.md). Bagian berikut mempertahankan peta versi sebelumnya.

**Tambahan setelah V4:** `analysis_v4/input_audit.py` dan [notebook 28](../notebooks/28_Audit_Input_Development_dan_Window_ECG.ipynb) mengaudit hanya development, raw versus window cache, kandidat QRS dan kasus hard/control. Artefak terpisah di `development_input_audit_20261003/` pada run V4; tidak mengubah paket training/fold/label. [Panduan fase](revisi/FASE_SETELAH_V4_20261003.md).

**Audit 28 selesai 230/230:** `analysis_v4/analyze_input_audit.py` memverifikasi sumber raw/tabel/kandidat dan menghasilkan analisis deskriptif serta hipotesis cakupan sampling di subfolder `analysis_20261003_complete/` milik audit input. [Laporan dasar](revisi/ANALISIS_AUDIT_INPUT_V4_20261003.md). Notebook hasil eksekusi dan kode audit input dipertahankan.

**V5 selesai 30/30 dan diaudit:** paket terpisah `research_v5/`, runner `notebooks/_v5_runner.py` dan notebook **29–32** dipertahankan. K8 belum memenuhi kriteria; baseline K4 tetap. `analysis_v5/audit_completed.py` memverifikasi 60 hasil/300 hash, exposure pasien dan pooling, kemudian menyimpan audit terpisah di `analysis_20261004_complete/` pada run V5. Tidak menimpa notebook/laporan asli atau melatih classifier. [Analisis V5](revisi/ANALISIS_HASIL_V5_20261004.md) · [panduan sebagai referensi](revisi/PANDUAN_MANUAL_V5.md).

**Revisi aktif 2 Oktober:** V3 **630 CV, 20 final dan 20 test**, notebook **18–19** dan SHAP Hierarchical V3 default 15 pasien selesai; **benchmark 17 juga selesai pada 75 rekaman INCART**. [Panduan](revisi/PANDUAN_MANUAL_V3.md) · [analisis classifier/SHAP](revisi/ANALISIS_HASIL_V3_20261002.md) · [analisis INCART](revisi/ANALISIS_BENCHMARK_INCART_20261002.md). `research_v3/` adalah paket training yang dibekukan. `analysis_v3/analyze_final.py` menghasilkan audit/CSV/grafik di `results/research_v3/20261001_patient_r_candidates_v3/analysis_20261002_complete/`. `analysis_v3/explain_final.py` menghasilkan SHAP primer V3 di `explain_primary_v3/729aa202d61f/`; `analysis_v3/audit_shap.py` memeriksa hash/sampel/array/ranking dan menyimpan `shap_v3_audit.json` beserta konteks prediksi segmen/pasien. `analysis_v2/explain_v2.py` menghasilkan SHAP V2 yang sudah selesai. `analysis_v3/analyze_incart.py` menyimpan audit/CSV/CI/grafik di `results/detector_benchmark/incart/848d6db41c5d/analysis_20261002/` tanpa menjalankan detector atau training. Snapshot pilot tetap di `analysis_20261001_pilot/`. Tabel historis di bawah menjelaskan pekerjaan sebelumnya.

Folder project: `/Users/daniandraaa/Desktop/TA GUWEH`.

## File alur historis V1

Jika menelusuri V1 dari awal, titik mulai adalah [notebook 00](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/00_Mulai_di_Sini.ipynb>). Training 00–04 sudah selesai; [notebook 05](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>) membaca hasil V1. Urutan aktif V3 tercantum pada [README utama](</Users/daniandraaa/Desktop/TA GUWEH/README.md>) dan [panduan menjalankan](</Users/daniandraaa/Desktop/TA GUWEH/README_REVISI.md>).

```text
TA GUWEH/
  README.md                         # buka ini untuk melihat urutan
  README_REVISI.md                   # petunjuk notebook dan Terminal
  notebooks/
    00_Mulai_di_Sini.ipynb           # persiapan dan GAN bersama
    01_Tuning_Encoder.ipynb          # 45 kasus CV Encoder
    02_Tuning_PatchTST.ipynb         # 45 kasus CV PatchTST
    03_Tuning_Hierarchical.ipynb     # 45 kasus CV Hierarchical
    04_Final_dan_Evaluasi.ipynb      # seleksi sampai laporan
    05_Visualisasi_Proses_dan_Hasil.ipynb  # 19 grafik; tanpa training
    _manual_runner.py               # dipanggil notebook untuk output/log
  configs/                          # protokol dan sembilan skenario
  src/                              # implementasi bersama
  scripts/                          # runner Terminal per tahap
  tests/                            # verifikasi implementasi
  docs/
    proposal/PROPOSAL TA.pdf
    referensi/Validasi Model Berbasis Pasien.pdf
    revisi/FASE_REVISI_FINE_TUNING.md
    revisi/REVISI_METODOLOGI_BAB_III.md
    revisi/ANALISIS_HASIL_EKSPERIMEN.md
    revisi/AUDIT_SKOR_09_DAN_ARAH_PUBLIKASI.md
    revisi/PRIORITAS_PERBAIKAN_DAN_EKSPERIMEN.md
  arsip/
    notebooks_lama/                 # EDA, baseline, fine-tuning historis
    debug/                          # patch_*, scratch_*, gambar percobaan
```

`_manual_runner.py` membantu menampilkan output dan menghentikan proses anak saat notebook di-interrupt. Notebook memanggil runner `scripts/` dan modul `src/` untuk training.

## Data, hasil, dan kode historis

| Lokasi | Fungsi |
|---|---|
| `Diagnosis.xlsx` | Metadata pasien dan diagnosis |
| `PVCVTRawECGData/` | Sinyal mentah |
| `PVCVTECGData/` | Sinyal dataset yang telah tersedia |
| `preprocessed_data/` | DWT, ID split, dan grafik preprocessing |
| `baseline_results/` | Checkpoint/metrik baseline historis |
| `finetune_results/` | Checkpoint/metrik fine-tuning historis |
| `results/finetune_revision/20260929_grid9_v1/` | Seluruh hasil eksperimen revisi aktif |
| `01_EDA_Preprocessing.py` | Kode preprocessing historis di root project |
| `02_Baseline_CNN_Transformer.py` | Kode baseline historis; digunakan verifikasi kompatibilitas model |
| `03_Finetune_CNN_Transformer.py` | Kode fine-tuning historis sebagai bukti eksperimen awal |

Data, folder hasil, dan tiga kode Python historis memakai lokasi yang telah dirujuk project. Notebook historis dipindahkan ke `arsip/notebooks_lama/` dengan isi dan output tersimpan tetap utuh. Untuk melihat hasil eksperimen lama, buka notebook arsip tersebut. Alur V1 memakai enam notebook 00–05; alur aktif V3/SHAP/benchmark memakai notebook 11–19.

## Letak keluaran revisi

Di dalam `results/finetune_revision/20260929_grid9_v1/`:

| Lokasi | Yang diperiksa |
|---|---|
| `protocol.json` | Grid dan identitas eksperimen yang dibekukan |
| `folds/` | Manifest pasien dan scaler fold |
| `augmentation/fold_1/` sampai `fold_5/` | Checkpoint/history GAN dan data sintetis |
| `encoder/`, `patchtst/`, `hierarchical/` | Hasil tuning, pilihan parameter, final, dan test per model |
| `baseline/` | Pembanding baseline dengan protokol revisi |
| `manual_logs/` | Log perintah dan output tiap tahap manual |
| `status.json` | Jumlah training CV/final/test yang lengkap |
| `comparison/` | CSV, grafik, dan `HASIL_REVISI.md` |
| `visualisasi/` | Sembilan belas PNG dari data sampai SHAP, termasuk audit GAN seluruh fold |

Jalankan sel status di notebook untuk memperbarui hitungan. Nama `pipeline.state` mencatat runner otomatis yang telah dihentikan; hitungan artefak tetap mencakup pekerjaan manual.

## Ekstensi V2 (30 September 2026)

- `research_v2/`: protokol, preprocessing fs=2000, ablation real/class-weight, seleksi dan laporan. Terpisah dari kode run historis.
- `notebooks/06_*` sampai `09_*`: urutan manual V2; `_research_runner.py` menyimpan log dan menangani interupsi.
- `results/research_v2/`: artefak baru; tidak menimpa run sebelumnya.
- `docs/revisi/PANDUAN_REVISI_V2.md`: alasan perubahan, urutan menjalankan, dan pekerjaan ilmiah yang belum selesai.
