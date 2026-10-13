# Analisis hasil notebook 11–17 — 1 Oktober 2026

**Dokumen ini snapshot pilot. Status 2 Oktober 2026: seluruh 630 CV, 20 final dan 20 test sudah selesai.** Baca [analisis lengkap V3](ANALISIS_HASIL_V3_20261002.md) untuk hasil terbaru; instruksi menyelesaikan 480 CV pada snapshot ini sudah tidak menjadi langkah aktif.

**Hasil sudah tersimpan dan audit lolos. V3 baru menyelesaikan pilot neural serta grid ExtraTrees: 150/630 kasus CV. Belum ada seleksi, model final atau skor test V3.** Seluruh sel notebook 11–17 memiliki execution count dan tidak menyimpan error, tetapi flag `False` menyebabkan full grid, final/test dan benchmark dilewati.

**Pembaruan setelah run ulang pada tanggal yang sama:** artefak tetap 150/630. Default notebook 12–14 kemudian diperbaiki ke full grid True; notebook 15 ke final/test True dengan guard kelengkapan CV. Notebook 12–15 dengan output lama diarsipkan di `arsip/notebooks_sebelum_fullgrid_20261001/`; output notebook aktif dibersihkan. Angka dan audit di laporan ini tetap merupakan hasil pilot sebelum perubahan alur, bukan hasil full grid baru.

## Bukti penyelesaian

| Tahap | Aktual | Yang masih diperlukan |
|---|---:|---|
| Persiapan kandidat development, notebook 11 | 230 pasien, 4.222 crop | Benchmark ketepatan detektor; anotasi PVC lokal belum tersedia |
| Encoder, notebook 12 | S01, 4 arm × 5 fold = 20/180 CV | S02–S09: 160 CV |
| CNN patch transformer, notebook 13 | 20/180 CV | S02–S09: 160 CV |
| Hierarchical, notebook 14 | 20/180 CV | S02–S09: 160 CV |
| ExtraTrees, notebook 15 | 90/90 CV | Seleksi setelah seluruh 630 CV lengkap |
| Training final/evaluasi V3 | 0/20 final; 0/20 test | Flag final/test masih `False`; test terkunci |
| SHAP Encoder final V2, notebook 16 | Selesai: 30 pasien background, 15 pasien dijelaskan | Analisis sensitivitas SHAP bila digunakan sebagai klaim interpretasi utama |
| Benchmark INCART, notebook 17 | Belum dijalankan | Ketiga flag install/download/run masih `False` |

Sisa CV adalah **480 kasus**, bukan pengulangan 630 kasus dari awal. Kasus S01 dan ExtraTrees yang sudah lengkap otomatis dilewati oleh runner. Jangan mengganti run-id atau protokol untuk meneruskan grid yang sama.

Audit menghitung ulang enam metrik dari seluruh 150 berkas probabilitas validation pasien. Semuanya cocok dengan `metrics.json`, dengan selisih maksimum **0** pada environment ini. Identitas kode/protokol V2–V3 valid; 46 pasien validation pada tiap fold cocok dengan manifest dan tidak masuk daftar training fold tersebut. Provenance sepuluh scaler sesuai pasien/segmen training. Hash raw development serta 230 shard kandidat cocok, dan array gabungan sama dengan shard. Audit ini memeriksa konsistensi artefak; tidak membuktikan validasi klinis atau mendeteksi semua kemungkinan kebocoran di luar manifest.

Fingerprint V3: `d93854c0b53937695ed86076bdc36249e7d12799796c5df9b7717a9af49d4dfb`.

## Hasil pilot neural

Semua nilai berikut adalah **mean macro-F1 patient-level ± SD lima fold**, konfigurasi **S01: LR 1e-4, batch 16**. Ini hasil development CV dengan early stopping, bukan hasil test dan bukan parameter terbaik dari grid lengkap.

| Model | Window 2,5 s, sampling segmen | Window 2,5 s, sampling pasien | Kandidat 0,5 s, sampling crop | Kandidat 0,5 s, sampling pasien |
|---|---:|---:|---:|---:|
| Encoder | 0,6364 ± 0,0946 | 0,6337 ± 0,0622 | 0,5746 ± 0,0431 | 0,5422 ± 0,0665 |
| CNN patch transformer | 0,6450 ± 0,0428 | 0,6287 ± 0,0525 | 0,5316 ± 0,0618 | 0,5093 ± 0,0628 |
| Hierarchical | 0,6497 ± 0,0826 | 0,6354 ± 0,0901 | 0,5441 ± 0,0525 | 0,5440 ± 0,0595 |

![Pilot CV](../../results/research_v3/20261001_patient_r_candidates_v3/analysis_20261001_pilot/pilot_cv.png)

Hierarchical window_segment memiliki mean tertinggi pada pilot ini. Selisihnya kecil terhadap Patch/Encoder dan grid neural baru menguji satu dari sembilan kombinasi, sehingga **belum boleh menetapkannya sebagai pemenang final**.

### Kandidat pendek belum memberi manfaat

Pada sampler pasien yang sama, delta kandidat minus window adalah −0,0915 untuk Encoder, −0,1194 untuk Patch, dan −0,0914 untuk Hierarchical. Setiap model turun pada **5/5 pasangan fold S01**. Dengan sampler segmen, kandidat turun pada 4/5 fold Encoder serta 5/5 fold Patch dan Hierarchical.

ExtraTrees menunjukkan arah serupa: **45/45 pasangan konfigurasi/fold** kandidat lebih rendah, mean delta −0,0585. Konfigurasi ExtraTrees terbaik dari grid comparator ialah **window_patient T09**, mean CV 0,6361 ± 0,0378; 500 pohon, max_depth 12, max_features 1,0, min_samples_leaf 4. Kandidat terbaik juga T09, 0,5712 ± 0,0575. Skor pemenang ini memiliki optimisme seleksi CV; tidak menjadi estimasi eksternal yang independen.

![Selisih pasangan](../../results/research_v3/20261001_patient_r_candidates_v3/analysis_20261001_pilot/matched_differences.png)

**Yang bisa disimpulkan:** representasi kandidat yang diuji belum meningkatkan klasifikasi tiga kelompok. **Penyebab belum terisolasi:** eksperimen ini sekaligus mengubah pusat potongan, durasi 2,5 menjadi 0,5 detik, jumlah/distribusi crop dan konteks waveform. Kandidat R/QRS juga belum tentu PVC. Kekurangan konteks, campuran jenis beat atau ketepatan detektor merupakan hipotesis untuk eksperimen berikutnya, bukan diagnosis yang sudah dibuktikan. Jangan menyebut semua penurunan sebagai kegagalan detector, label salah, atau LR yang salah.

### Sampling pasien mengubah distribusi kesalahan

Pada window 2,5 detik, delta patient minus segment sampler adalah −0,0027 Encoder, −0,0162 Patch dan −0,0143 Hierarchical. Arah fold tidak seragam. Sampling pasien menyamakan kontribusi pasien dalam ekspektasi; hal itu tidak otomatis menaikkan macro-F1.

Contoh Encoder S01 pada pooled OOF (230 pasien): recall RVOT cusp naik **54/103 → 74/103**, tetapi recall RVOT septal/free-wall/other turun **47/70 → 27/70**. Ini pertukaran kesalahan antarkelas yang perlu dibahas, bukan peningkatan menyeluruh. Pooled OOF macro-F1 berbeda secara definisi dari mean macro-F1 fold; jangan menukar kedua angka di tabel laporan.

![Recall kelas](../../results/research_v3/20261001_patient_r_candidates_v3/analysis_20261001_pilot/oof_class_recall.png)

### Training dan batas hasil

Loss training turun pada semua kelompok pilot, sementara F1 validation berfluktuasi. Median best epoch Encoder window_patient adalah 1 dan kandidat_patient 3; median kandidat_patient Patch/Hierarchical juga 3. Early stopping menyimpan skor terbaik; audit memastikan skor dan best epoch cocok dengan history. Menambah epoch secara langsung belum didukung pola ini. Loss training dan macro-F1 validation mengukur hal berbeda dan tidak membuktikan penyebab penurunan.

Perbandingan terhadap V2 tidak mengisolasi satu perbaikan: V3 mengubah class weights menjadi frekuensi **pasien**, sampling dengan pengembalian, serta penanganan tail patch pada Patch. Perbandingan kandidat/window dan sampler di dalam V3 dengan scenario/fold sama lebih informatif. Belum ada dasar untuk menyebut kinerja sekarang sudah maksimal.

## Kandidat dan SHAP

Kandidat mencakup seluruh 230 pasien development, **tanpa fallback**. Jumlah crop per pasien 2–64, median 14. Ada **16/230 pasien (7,0%)** dengan engineering quality flag. Median agreement dua sadapan 0,973. Agreement adalah kesesuaian detector antarsadapan, bukan sensitivitas terhadap anotasi PVC atau jaminan kualitas klinis. Tidak ada pasien dibuang dan tidak ada diagnosis diubah.

Notebook 16 benar-benar menghasilkan SHAP **Encoder V2 seed 42**, bukan V3. Audit memastikan checkpoint, kode, signature, daftar pasien, 15 array individual, array gabungan `(15,12,5000)` dan importance konsisten serta semua nilai finite. Background 30 pasien berasal dari development; explain set 15 pasien historical test terdiri dari 5 pasien tiap kelas, satu segmen tengah per pasien, `nsamples=100`.

Mean absolute SHAP tertinggi pada explain set ini adalah aVR (0,002045), V4 (0,002008), V5 (0,001876). Target ialah **true-class segment logit**. Urutan ini tidak membuktikan sadapan tersebut paling penting secara klinis, tidak menjelaskan seluruh agregasi probabilitas pasien, dan tidak cukup stabil untuk klaim fisiologi tanpa analisis background/sampling serta validasi tambahan.

Benchmark INCART belum menghasilkan metrik. Notebook 17 berhasil mengeksekusi sel yang mencetak instruksi karena flag masih `False`; ini tidak sama dengan benchmark selesai.

## Urutan manual berikutnya

1. Buka ulang notebook **12**, **Restart Kernel & Run All**. Default `RUN_FULL_GRID = True` sudah disiapkan; pastikan output `Mode: FULL GRID 180 fit`. Runner menggunakan S01 yang selesai dan mengerjakan 160 kasus tersisa. Jalankan satu training MPS pada satu waktu.
2. Lakukan hal yang sama pada notebook **13**, lalu **14**. Tetap gunakan run-id, LR `[1e-4,5e-4,1e-3]`, batch `[16,32,64]`, empat arm dan fold yang sudah dibekukan.
3. Jalankan **notebook 15**. Default `RUN_FINAL_AND_TEST = True` sudah disiapkan; pemeriksaan setiap kasus CV harus **630/630** sebelum seleksi/final/test dilanjutkan. ExtraTrees yang selesai dilewati; program memilih dari CV, melatih 4 model × 5 seed, kemudian mengevaluasi 59 pasien historical test tanpa k-fold. Seed 42 tetap primer.
4. Untuk bukti ketepatan detector, jalankan notebook **17** dengan `INSTALL_WFDB=True`, `DOWNLOAD_DATA=True`, `RUN_BENCHMARK=True` pada penggunaan pertama. Setelah dependensi/data tersedia, install/download dapat dikembalikan ke False. Benchmark deteksi beat publik tidak menggantikan validasi eksternal tiga kelompok anatomi atau anotasi PVC lokal.
5. Setelah CV/final lengkap, analisis hasil semua scenario dan seed sebelum merancang eksperimen baru. Kandidat dapat dilaporkan sebagai ablation yang tidak membantu; jangan menghapus arm karena nilainya rendah. Perubahan durasi, detector, loss/arsitektur atau agregasi harus memakai protokol/run baru dan dirancang menggunakan development.

**Untuk target publikasi:** prioritas sekarang ialah menuntaskan bukti ablation yang sudah ditetapkan, benchmark detector dan pelaporan hasil/keterbatasan. Neural tuning lengkap, GAN terkoreksi jika diwajibkan objektif proposal, nested/repeated CV dan validasi pasien baru dengan label anatomi masih belum selesai. Hasil historical test yang telah terlihat tidak berubah menjadi test independen dengan membuat run baru. SHAP dan angka tinggi saja tidak menjamin publikasi Q1.

Artefak audit: `results/research_v3/20261001_patient_r_candidates_v3/analysis_20261001_pilot/` berisi JSON audit, daftar 480 kasus yang belum selesai, metrik hasil hitung ulang, pooled OOF/recall, pasangan fold, status notebook dan tiga grafik. Program audit dapat dijalankan ulang tanpa training:

```bash
.venv/bin/python -m analysis_v3.analyze_completed
```

Audit menulis turunan analisis saja; paket training V1/V2/V3 dan identitas protokol tidak diubah. Berkas `verification.json` sebelumnya tetap merupakan catatan uji fixture sebelum training nyata; hasil audit terbaru ada di `analysis_20261001_pilot/audit.json`.
