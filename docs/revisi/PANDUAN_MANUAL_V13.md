# Panduan manual V13 — Opsi A: baseline vs fine-tuning sesuai proposal

Disiapkan 7 Oktober 2026. V13 menjalankan ulang desain proposal (Bab III) dengan perbaikan dari audit V1–V12. Run belum dijalankan. Hasil lama tidak diubah.

## Apa yang berbeda dari run V1 (`20260929_grid9_v1`)

| Masalah di V1 | Perbaikan di V13 |
|---|---|
| Sampling rate tercatat 1.000 Hz | Cache native 2.000 Hz (V2 dan seterusnya), window 2,5 detik |
| Label "Non-OT" | Label anatomi terkoreksi: LVOT / RVOT cusp / RVOT septal-free-wall-other (anggota pasien sama) |
| GAN kasar 10–55× dan relasi sadapan rusak | GAN V7 per fold yang lolos QC; GAN final baru dengan resep V7 + QC yang sama |
| Resep baseline berbeda dari fine-tuning (AdamW + class weight vs Adam tanpa class weight) | Resep identik; yang berbeda hanya **GAN + Grid Search** |
| Satu seed | CV 5 fold × **3 seed**; final **5 seed** |
| Tanpa uji signifikansi | Wilcoxon (CV), Friedman (antar-model), McNemar exact + bootstrap CI (test), koreksi Holm |
| SHAP hanya model utama | SHAP keenam model final (3 arsitektur × 2 skenario) |

## Desain

- **Data:** 230 pasien development (LVOT 57, RVOT cusp 103, RVOT other 70) dan 59 pasien test historis. Input berupa sinyal 12 sadapan yang sudah di-denoise dengan DWT, 2.000 Hz, window 2,5 detik.
- **Model:** CNN–Transformer Encoder, CNN–PatchTST, CNN–Hierarchical Transformer. Klasifikasi tingkat pasien dengan mean pooling atas window.
- **Resep bersama:** AdamW, weight decay 1e-4, maksimum 100 epoch, early stopping patience 15 pada macro-F1 validasi, cosine annealing (T_max 100, LR minimum 1e-6), gradient clipping 1,0, cross-entropy berbobot kelas (inverse frequency pasien training), 4 window per pasien per epoch.
- **Skenario 1 – Baseline:** LR 1e-3, batch 32 segmen (8 pasien), tanpa GAN dan tanpa grid.
- **Skenario 2 – Fine-tuning:** Grid Search LR {1e-4, 5e-4, 1e-3} × batch {16, 32, 64} (G01–G09) dengan augmentasi GAN. Pada setiap bag training penuh, 1 dari 4 window diganti window sintetis dari kelas yang sama. Validasi dan test **selalu data asli**. GAN tiap fold hanya dilatih dari pasien training fold tersebut.
- **Isolasi efek GAN:** konfigurasi G08 (LR 1e-3, batch 32) sama dengan baseline. Perbandingan baseline vs G08 menunjukkan efek GAN saja.
- **Seleksi:** konfigurasi fine-tuning terbaik per model dipilih dari mean macro-F1 15 kasus CV. Epoch final = median best epoch. Test tidak dipakai untuk memilih apa pun.
- **Final:** 3 model × 2 skenario × 5 seed (42–46) dilatih pada seluruh 230 pasien. Model fine-tuning memakai GAN final.
- **Test:** dibuka hanya setelah seleksi, QC GAN final, dan 30 model final lengkap.

## Urutan menjalankan

Buka notebook di Jupyter/VS Code, pilih kernel `.venv` project, lalu jalankan sel satu per satu (Shift+Enter). Setiap tahap bisa dilanjutkan bila terputus: jalankan ulang sel yang sama.

| Notebook | Pekerjaan | Jumlah fit | Estimasi MPS |
|---|---|---:|---|
| [69](../../notebooks/69_V13_Persiapan_dan_Smoke.ipynb) | Persiapan scaler, bahan GAN final, smoke | 0 (6 update sekali pakai) | < 1 menit |
| [70](../../notebooks/70_V13_CV_Encoder.ipynb) | CV Encoder: baseline + 9 grid | 150 | ±5 jam |
| [71](../../notebooks/71_V13_CV_PatchTST.ipynb) | CV PatchTST | 150 | ±2 jam |
| [72](../../notebooks/72_V13_CV_Hierarchical.ipynb) | CV Hierarchical | 150 | ±4–5 jam |
| [73](../../notebooks/73_V13_Seleksi_GAN_Final_dan_Model_Final.ipynb) | Seleksi → GAN final (100 epoch) → QC → 30 final | 30 + 1 GAN | ±2–3 jam |
| [74](../../notebooks/74_V13_Test_dan_Analisis_Bab_IV.ipynb) | Test 20% + statistik + grafik Bab IV | 0 | menit |
| [75](../../notebooks/75_V13_SHAP_Enam_Model.ipynb) | SHAP 6 model (CPU) | 0 | ±30–60 menit |
| [76](../../notebooks/76_V13_Analisis_Sekunder_Biner_dan_Ensemble.ipynb) | Analisis sekunder pra-registrasi: biner LVOT vs RVOT, ensemble 5 seed ([pra-registrasi](PRAREGISTRASI_ANALISIS_SEKUNDER_V13.md)) | 0 | menit |

Estimasi waktu berdasarkan durasi fit V4/V8 (±0,6–2 menit per fit di MPS). Notebook 70–72 bisa dijalankan di hari yang berbeda.

Lewat Terminal juga bisa, misalnya:

```bash
.venv/bin/python -m research_v13 status --run-id 20261008_proposal_baseline_vs_finetune_v13
```

**Ruang disk:** run penuh membutuhkan ±0,5 GB (CV ±50 MB, 30 model final ±300 MB, GAN ±50 MB). Protokol berhenti bila disk bebas < 1,75 GiB. Saat disiapkan disk hanya tersisa ±2,7 GiB dan terus berkurang; sebaiknya kosongkan beberapa GB sebelum mulai. Bila muncul pesan disk, kosongkan ruang lalu jalankan ulang sel yang sama.

## Keluaran untuk Bab IV

Folder `results/research_v13/20261008_proposal_baseline_vs_finetune_v13/reports/`:

- `cv_summary.csv`: mean ± SD macro-F1, akurasi, sensitivitas, spesifisitas dan AUC CV untuk tiap model × skenario (tabel hasil Grid Search).
- `bab4_tabel_test.csv`: lima metrik test (mean ± SD 5 seed, ditambah seed 42).
- `test_per_class.csv`: precision/recall/F1 tiap kelas.
- `stat_cv_wilcoxon.csv`, `stat_cv_friedman_models.csv`, `stat_test_mcnemar_bootstrap.csv`: uji signifikansi.
- `RINGKASAN_V13.md` dan grafik: boxplot CV, heatmap grid, sebaran test per seed, confusion matrix, recall per kelas.
- `../shap/lead_importance.csv|png`: kontribusi tiap sadapan untuk keenam model.

## Batas interpretasi

- 59 pasien test pernah dipakai pada eksperimen V1–V4. Sebut sebagai **hold-out historis**, bukan validasi eksternal.
- Skor CV fine-tuning bersifat optimistis karena konfigurasinya dipilih dari CV yang sama (bukan nested CV). Perbandingan yang adil antar-skenario ada di **test**.
- Pasangan fold × seed tidak sepenuhnya independen. Wilcoxon CV dipakai sebagai bukti pendukung; uji utama adalah McNemar + bootstrap di test.
- QC GAN adalah skrining teknis, bukan validasi klinis morfologi.
- "Fine-tuning" di sini berarti optimasi hiperparameter + augmentasi GAN dari bobot acak, bukan transfer learning. Jelaskan definisi ini di Bab III.
- Hasil negatif (misalnya fine-tuning tidak lebih baik pada salah satu model) tetap dilaporkan.
