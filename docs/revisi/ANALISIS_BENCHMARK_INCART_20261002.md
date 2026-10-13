# Analisis benchmark INCART — 2 Oktober 2026

**Notebook 17 sudah selesai menjalankan benchmark nyata:** 5/5 sel, tanpa error tersimpan, seluruh I01–I75 tersedia dan memiliki hasil. Folder hasil: `results/detector_benchmark/incart/848d6db41c5d/`. Dependency WFDB 4.3.1 dipasang pengguna di target QA terpisah `.ecgqa/`; paket training V2/V3 tetap sesuai identitas protokol.

**F1 0,9686 ini adalah deteksi posisi beat pada INCART.** Ia tidak menggantikan macro-F1 klasifikasi tiga kelompok asal anatomi. Hasil historical test primer Hierarchical V3 tetap 0,5837; rata-rata lima seed 0,5783 ± 0,0783. Analisis ini tidak menjalankan training classifier, mengubah label lokal, atau memilih ulang model.

## Hasil utama

| Ukuran | Hasil | 95% CI bootstrap sumber Holter |
|---|---:|---:|
| Precision semua beat | 95,41% | 92,84–97,50% |
| Recall semua beat | 98,36% | 96,60–99,57% |
| F1 semua beat, micro | 0,96862 | 0,95373–0,98093 |
| Recall deteksi beat beranotasi V | 92,79% | 84,42–98,33% |
| Recall beat dengan simbol selain V | 99,07% | 97,34–99,94% |

Reference yang memenuhi aturan crop lengkap: **175.867 beat**. TP **172.981**, FP **8.322**, FN **2.886**. Beat V reference **20.008**, terdeteksi **18.566**, terlewat **1.442**. Beat V mencakup **49,97% seluruh FN**, meskipun hanya sekitar 11,38% eligible reference. Simbol selain V mencakup beberapa jenis beat; istilah ini tidak berarti semuanya sinus normal.

F1 rata-rata per rekaman **0,96299**, median **0,99335**, minimum **0,45185**. Mean per-record dan F1 micro dari total hitungan merupakan statistik berbeda. Lima rekaman tanpa reference V (I34/I60/I61/I70/I71) memiliki recall V tidak terdefinisi; nilai kosong tidak diubah menjadi nol atau seratus persen.

INCART memiliki 75 cuplikan dari **32 sumber Holter**; nomor pasien dalam header menghubungkan cuplikan dari sumber yang sama. Sampling 257 Hz dan anotasi yang waktunya sesekali kurang selaras juga dijelaskan oleh [deskripsi resmi PhysioNet](https://physionet.org/content/incartdb/1.0.0/). Audit membaca 32 grup langsung dari header. Bootstrap 5.000 draw, seed 20261002, mengambil ulang **grup sumber lengkap**; ia tidak memperlakukan setiap beat atau 75 cuplikan sebagai pasien independen. CI bersyarat pada detector tetap dan kohort yang diamati; tidak mencakup seluruh ketidakpastian revisi algoritme atau perubahan domain.

## Kegagalan yang harus dipertahankan dalam pelaporan

| Rekaman | Temuan utama | Bukti |
|---|---|---|
| I70 | Banyak beat terlewat | Recall 29,29%; F1 0,4519; FN 1.178 |
| I05 | Banyak deteksi tambahan | Precision 53,93%; FP 1.514; F1 0,7001 |
| I26 | Banyak deteksi tambahan | Precision 54,24%; FP 1.273; F1 0,7033 |
| I59 | Recall V rendah | 21/81 = 25,93% |
| I67 | Recall V rendah | 180/531 = 33,90% |
| I65 | Recall V rendah | 150/382 = 39,27% |
| I18 | Recall V rendah | 182/366 = 49,73% |
| I72 | Recall V rendah | 244/386 = 63,21% |
| I62 | Recall V lebih rendah dari agregat | 623/795 = 78,36% |

I05/I26 menyumbang **33,49% seluruh FP**; I70 menyumbang **40,82% seluruh FN**. Enam rekaman V dengan support minimal 50 dan recall di bawah 80% (I59/I67/I65/I18/I72/I62) mencakup **79,13% V yang terlewat**. Batas support/recall tersebut hanya membantu mendeskripsikan antrean kegagalan; tidak digunakan untuk eksklusi atau tuning. Seluruh 75 rekaman tetap masuk hasil utama, termasuk satu reference V pada I07 yang tidak terdeteksi.

Agreement antar-sadapan yang tinggi bukan ukuran akurasi terhadap reference: agreement I05 0,9712, tetapi precision hanya 0,5393. I59 juga memiliki agreement 0,9081 ketika recall V hanya 0,2593. Maka QC agreement lokal yang tinggi tidak cukup untuk menyatakan kandidat PVC tervalidasi. Kegagalan ini belum menetapkan penyebab fisiologis atau kesalahan label; penyebab spesifik perlu analisis engineering dengan sinyal/reference.

![Hasil dan kegagalan benchmark](../../results/detector_benchmark/incart/848d6db41c5d/analysis_20261002/benchmark_overview.png)

## Apa yang sudah diverifikasi

Endpoint baru `analysis_v3.analyze_incart` hanya membaca hasil dan menulis analisis. Audit lolos untuk:

- Signature protokol/folder, hash kode detector/benchmark, spesifikasi tetap serta **225 hash file** `.hea/.dat/.atr`.
- Seluruh 75 JSON per rekaman dan CSV, 12 sadapan/header 257 Hz, 32 grup sumber, jumlah eligible beat/reference V dan eksklusi tepi dari anotasi asli.
- Precision/recall/F1 serta recall V tiap rekaman dihitung ulang dari hitungan tersimpan; selisih maksimum **2,22 × 10⁻¹⁶** karena pembulatan numerik. Summary micro cocok dengan total TP/FP/FN.
- Empat pengujian bootstrap: pooling berbasis hitungan, satu grup dengan hasil tetap, recall V tidak terdefinisi tanpa V, dan penolakan hitungan yang mustahil.

**40 reference beat di tepi** tidak memenuhi crop lengkap. Tujuh di antaranya memiliki indeks negatif pada anotasi sumber (I04/I17/I35/I44/I57/I72/I74), termasuk dua V. Aturan eligibility benchmark sudah mengecualikannya; audit mempertahankan anotasi asli dan memeriksa jumlah eksklusi, tanpa mengubah reference.

Audit ini **tidak menjalankan ulang detector atau matching dari waveform**, karena indeks peak/pasangan lengkap tidak disimpan pada run ini. Konsistensi hitungan/hash bukan verifikasi ulang seluruh TP/FP/FN dari inference. Median dari median timing rekaman adalah 3,89 ms; ini bukan median gabungan semua pasangan. Toleransi nominal 100 ms dibulatkan menjadi 26 sampel pada 257 Hz, sekitar 101,17 ms.

## Batas penggunaan hasil

1. Recall V menghitung lokasi beat beranotasi V yang tertangkap oleh detector semua beat. Detector tidak mengeluarkan label sinus/PVC; **precision/F1 diagnosis PVC belum diukur**.
2. Benchmark tidak memvalidasi kelompok LVOT/RVOT cusp/RVOT septal-free-wall-other; tidak ada pelatihan atau evaluasi classifier asal anatomi di INCART.
3. Sampling/domain benchmark berbeda dari data lokal 2.000 Hz. Aturan waktu detector sama dalam detik, tetapi hasil ini tidak menjadi akurasi beat pada dataset lokal tanpa anotasi reference lokal.
4. Batas 64 crop dinonaktifkan pada benchmark agar semua peak dapat dinilai. Recall ini bukan proporsi seluruh V yang masuk ke 64 crop training lokal. Kandidat lokal tetap bukan PVC terverifikasi.
5. Threshold tidak dicari pada benchmark. Jika detector direvisi menggunakan kegagalan INCART yang kini terlihat, simpan versi/protokol baru dan perlakukan INCART sebagai data pengembangan untuk revisi tersebut. Validasi revisi memerlukan data anotasi lain yang belum digunakan untuk pemilihan algoritme.

## Langkah berikutnya

**Tahap menjalankan V3 saat ini sudah lengkap:** training/final/test 12–15, SHAP V2 16, benchmark 17, analisis 18, SHAP primer V3 19. Tidak perlu mengulangnya untuk mendapatkan skor yang berbeda.

Prioritas classifier adalah **stabilitas training dan pemisahan dua kelompok RVOT**. Mulai protokol baru pada development dengan kontrol window 2,5 detik dan comparator yang sama, evaluasi variasi seed pada validation/OOF, lalu pilih recipe berdasarkan CV. Bila diperlukan estimasi yang mengurangi optimisme pemilihan hyperparameter, gunakan nested patient CV; menjalankan ulang model terpilih pada development yang sama tidak otomatis menghasilkan estimasi independen. Test historis 59 pasien tetap sudah pernah diamati.

Prioritas representasi adalah **kontrol durasi dan pusat crop**: perbandingan kandidat 0,5 detik dengan window 2,5 detik saat ini mengubah beberapa komponen sekaligus. Benchmark yang baik pada sebagian besar beat tidak membuktikan crop pendek optimal untuk asal anatomi. Detector pembanding/revisi dapat dikerjakan dalam versi baru; simpan peak/pasangan agar kegagalan dapat diperiksa ulang, tanpa mengubah run V3 yang sudah selesai.

Untuk menuntaskan TA, masukkan hasil negatif ablation, kegagalan detector, variasi seed, CI dan batas SHAP ke Bab IV–V. GAN terkoreksi pada preprocessing 2.000 Hz beserta ablation tetap pending bila diwajibkan objektif proposal; kerjakan dengan data training fold saja dan recipe classifier yang sebanding. Kontribusi ilmiah dan validasi pasien baru tetap perlu diperkuat untuk target publikasi; skor deteksi beat ini tidak menjamin penerimaan Q1.

Artefak analisis baru: `analysis_20261002/audit.json`, `record_diagnostics.csv`, `source_holter_counts.csv`, `records_by_f1.csv`, `records_by_V_recall.csv`, `source_holter_bootstrap.csv`, `benchmark_overview.png` di folder benchmark. File hasil asli dan output notebook pengguna dipertahankan.

Untuk membaca ulang audit tanpa training, download atau menghitung detector:

```bash
.venv/bin/python -u -m analysis_v3.analyze_incart --benchmark results/detector_benchmark/incart/848d6db41c5d
```
