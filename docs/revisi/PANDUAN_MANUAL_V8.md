# Menjalankan pembanding classifier V8

**Status 6 Oktober 2026: notebook 41–45 telah selesai dan seluruh 120 fit terverifikasi. Tidak perlu menjalankan ulang training.** [Analisis hasil](ANALISIS_HASIL_V8_20261006.md) · [fase berikutnya](FASE_SETELAH_V8_20261006.md). Instruksi di bawah menjadi referensi alur yang sudah diselesaikan.

**Persiapan kelima fold telah selesai tanpa training classifier.** Sel `prepare` pada 41 akan memverifikasi hasil yang tersedia; smoke MPS belum dijalankan. Tujuh tes CPU sintetis lulus dan 350 file historis tetap utuh. [Audit implementasi](../../results/research_v8/20261005_paired_classifier_augmentation_v8/implementation_audit.json).

**Mulai notebook 41 → 42 dahulu.** Setelah pilot selesai tanpa masalah proses, lanjutkan 43 → 44 → 45. GAN V7 sudah lengkap/PASS; notebook 37–40 tidak perlu diulang.

Fase ini menguji manfaat augmentasi pada **macro-F1 pasien**, bukan sekadar mutu sinyal. Seluruh training dan smoke MPS data asli tetap Anda jalankan manual menggunakan kernel `.venv` project.

| Urutan | Notebook | Proses |
|---|---|---|
| 1 | [41 — Persiapan dan smoke](../../notebooks/41_Persiapan_dan_Smoke_Classifier_Augmentasi_V8.ipynb) | Verifikasi sumber, scaler inner/outer-training, lalu smoke kedua model/tiga arm |
| 2 | [42 — Pilot fold 1](../../notebooks/42_Pilot_Fold1_Classifier_Tiga_Arm_V8.ipynb) | Dua selector inner + enam pembanding, seed 42, kemudian analisis pilot |
| 3 | [43 — Hierarchical](../../notebooks/43_Pembanding_Hierarchical_Lima_Fold_V8.ipynb) | Lima fold/tiga seed; kasus pilot lengkap digunakan ulang |
| 4 | [44 — Encoder](../../notebooks/44_Pembanding_Encoder_Lima_Fold_V8.ipynb) | Lima fold/tiga seed; kasus pilot lengkap digunakan ulang |
| 5 | [45 — Analisis lengkap](../../notebooks/45_Analisis_Manfaat_GAN_Classifier_V8.ipynb) | Audit prediksi, efek berpasangan, OOF, recall kedua RVOT, variasi seed dan biaya |

Run: `20261005_paired_classifier_augmentation_v8`. Kode terpisah: `research_v8/`; runner: `_v8_runner.py`; hasil: `results/research_v8/20261005_paired_classifier_augmentation_v8/`.

## Cara menjalankan

1. Pilih interpreter/kernel `.venv/bin/python` dari project di VS Code atau Jupyter.
2. Buka 41, jalankan setup dan `prepare`. Persiapan hanya membuat/verifikasi scaler dan manifest; tidak melatih classifier. Jika hasil sudah tersedia, verifikasi dan gunakan ulang.
3. Jalankan smoke MPS. Enam update disposable menguji dua model × tiga arm pada batch 8 pasien × maksimal 4 window native. Bobot smoke dibuang; tidak digunakan sebagai awal training.
4. Jika smoke lulus, buka 42 dan jalankan pilot. Log menjelaskan `inner_selection` lalu `real_only`, `real_aug`, `gan_aug` untuk setiap model. Jalankan analisis pilot dan status sampai selesai.
5. Setelah pilot proses selesai, jalankan 43 dan 44 secara bergantian; satu proses training GPU pada satu waktu. **Hasil pilot F1 rendah/negatif bukan alasan memilih hanya arm atau seed tertentu.** Jangan memilih rasio dari satu fold.
6. Buka 45 setelah status menunjukkan `inner_selection_complete: 30` dan `outer_comparison_complete: 90`. Setelah analisis selesai, minta audit hasil dan keputusan berikutnya.

Interrupt/Stop menghentikan child process. Tunggu proses berhenti, lalu ulang sel yang sama pada device yang sama. Resume memakai epoch lengkap terakhir; epoch terputus diulang. Checkpoint memuat optimizer/scheduler, RNG global/dropout, loader, crop, slot dan sumber augmentasi. Hasil lengkap tidak ditrain ulang dan harus lolos verifikasi hash.

Jika smoke atau training gagal, periksa ekor log yang tampil. Jangan otomatis menurunkan batch, mengubah rasio, mengedit epoch atau mengganti device di tengah run. Perubahan protokol/runtime membutuhkan run baru. Disk guard membutuhkan minimal 3 GiB ditambah cadangan; tidak menghapus hasil lama atau menggandakan cache waveform.

## Bagaimana pembanding dibuat setara

- **Real-only:** bag berisi sampai empat window dari pasien anchor.
- **Real augmentation:** pada bag penuh K4, satu slot dipilih acak dan diganti window real dari pasien lain dengan kelas sama. Donor dipilih uniform pasien GAN-fit kelas itu, lalu uniform window; anchor tidak menjadi donor dirinya sendiri.
- **GAN augmentation:** pada slot yang sama, satu window diganti dari bank V7 kelas yang sama dan fold yang sesuai. Bank yang sudah lolos QC berisi 64 window per kelas. Tidak membuat sintetis baru, melakukan smoothing atau memilih waveform terbaik.

Semua arm memakai urutan pasien anchor dan crop asli yang sama, seed/inisialisasi model sama, mask/panjang bag sama dan jumlah update sama. RNG crop, slot dan pemilihan sumber terpisah. Perbedaan slot yang diganti tidak menggeser crop/dropout arm lain.

Penggantian satu window hanya berlaku pada **bag empat window**, sehingga dosisnya 25% pada bag yang eligible. **22 dari 230 pasien development mempunyai kurang dari empat window**; mereka tetap dilatih dan dinilai, dengan bag asli tanpa penggantian. Tidak membuang pasien atau memperbesar porsi sintetis pada bag pendek. Proporsi seluruh window yang diganti akan sedikit di bawah 25%, dicatat pada history.

Mixed bag merupakan input training artificial, bukan rekaman seorang pasien baru. Loss tetap diberi label pasien anchor; donor/sintetis memiliki kondisi diagnosis kelas sama. Asumsi ini diuji melalui kontrol real augmentation, belum validasi klinis PVC/anatomi sintetis.

## Pemilihan epoch dan pemisahan pasien

Setiap model/fold/seed terlebih dahulu melatih **real-only pada pasien GAN-fit**, dengan scaler hanya fit di kelompok tersebut. Sebanyak 37 pasien referensi V7 digunakan sebagai **inner validation classifier**, dengan data real dan seluruh window pasien, untuk memilih epoch macro-F1 tertinggi pertama. Batas 100 epoch dan patience 15; selector tidak memakai augmentation atau outer validation.

Setelah itu, ketiga arm dimulai kembali dari bobot acak pada seluruh **outer-training = GAN-fit + inner-reference**. Scaler outer-training disalin persis dari V4, dengan provenance/hash yang diperiksa. Semua arm memakai **jumlah epoch pilihan inner yang sama**, AdamW, cosine T_max 100, LR dan pooling model yang sama. Tidak memakai bobot selector sebagai bobot pretrained.

Outer validation 45–47 pasien hanya diprediksi **setelah** training fixed budget selesai. Seluruh window real dipool secara global; tidak memilih epoch dari outer F1 dan tidak memakai sintetis di evaluation.

LR/pooling tetap mengikuti pilihan development V4: **Hierarchical mean, LR 5e-4; Encoder attention, LR 1e-4**. Batch pasien 8, K4, weight decay 1e-4, clip 1 dan inverse patient class-frequency weighting. Karena pilihan LR/pooling dan desain sebelumnya telah melihat development, V8 tetap **ablation eksploratif**, bukan estimasi nested yang bebas seluruh pemilihan sebelumnya. Test historis tidak dibuka pada V8.

## Anggaran dan output

**30 selector inner + 90 pembanding outer = 120 fit**: dua model × lima fold × tiga seed × (satu selector + tiga arm). Pilot delapan fit sudah termasuk anggaran ini; per model, 43/44 melengkapi 56 fit setelah empat kasus pilotnya. Kontrol classifier V4/V5 tidak digunakan ulang karena pemilihan epoch dan recipe evaluasinya berbeda. Generator V7 tetap digunakan ulang tanpa training GAN.

Pada setiap case: `case.json`, `history.csv`, `metrics.json`, `predictions.csv`, `segment_weights.csv`, checkpoint dan manifest hash. History mencatat anchor/kelas, eligible bag, jumlah penggantian, donor real, indeks sintetis, update, gradient norm dan waktu.

Laporan `analysis_complete/` menyediakan `cases.csv`, `paired_deltas.csv`, `seed_summary.csv`, `pooled_oof.csv`, `patient_predictions.csv`, grafik dan `REPORT.md`. Macro-F1, recall, coverage dan pooling diperiksa dari prediksi tersimpan. Mean/SD fold/seed adalah deskriptif; jangan memperlakukan 15 hasil sebagai pasien independen atau memilih seed terbaik.

Jika GAN membantu secara konsisten terhadap **kedua kontrol**, lanjutkan ablation komponen/seed generator dan perluasan arsitektur proposal. Jika tidak, laporkan hasil negatif dan prioritaskan alternatif yang sesuai data. Belum ada skor classifier asli V8; skor 0,9 dan Q1 tidak dijanjikan.

Alternatif Terminal dari root project:

```bash
.venv/bin/python -m research_v8 prepare
.venv/bin/python -m research_v8 smoke --device mps
.venv/bin/python -m research_v8 pilot --device mps
.venv/bin/python -m research_v8 analyze --pilot-report
.venv/bin/python -m research_v8 train --models hierarchical --device mps
.venv/bin/python -m research_v8 train --models encoder --device mps
.venv/bin/python -m research_v8 analyze
.venv/bin/python -m research_v8 status
```

Pengujian otomatis memakai sinyal sintetis CPU: isolasi donor, short bag, crop/slot berpasangan, gradien model native, inner/outer selection, resume identik termasuk recovery history, seluruh OOF lima fold dan notebook valid. Smoke MPS data asli perlu dikonfirmasi pada 41; tes CPU tidak menggantikannya.
