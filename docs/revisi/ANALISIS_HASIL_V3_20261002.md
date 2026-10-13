# Analisis hasil lengkap V3 — 2 Oktober 2026

**Bagian tuning dan evaluasi V3 sudah selesai:** 630/630 CV, 20/20 checkpoint final, 20/20 evaluasi historical test. Seluruh sel notebook 12–15 selesai dan tidak menyimpan error. Audit independen menghitung ulang enam metrik dari 630 prediksi validation dan 20 prediksi test; semuanya cocok dengan selisih maksimum **0** pada environment ini. Protokol/fingerprint V2–V3 tetap valid.

**Model primer V3 adalah Hierarchical, window_segment S04, LR 5e-4, batch 16, 9 epoch, seed 42.** Dipilih dari development CV, bukan dari hasil test. Patch memiliki test seed 42 tertinggi, tetapi tidak menggantikan pilihan primer setelah hasil test terlihat. Tiga model neural dan satu comparator masing-masing memiliki lima checkpoint, seed 42–46; tidak ada ensemble otomatis.

**Pemeriksaan run manual berikutnya:** notebook 18 selesai 7/7 sel dan notebook 19 selesai 3/3 sel tanpa error tersimpan. Analisis lengkap berhasil diperbarui dan SHAP model primer V3 default **15/15 pasien** sudah terbentuk serta lolos audit artefak. **Notebook 17 juga sudah selesai 5/5 sel dengan benchmark 75 rekaman INCART**: F1 deteksi beat 0,96862 dan recall V 92,79%; [laporan dan batas penggunaan](ANALISIS_BENCHMARK_INCART_20261002.md). Tidak ada training baru atau perubahan skor classifier dari analisis/SHAP/benchmark ini.

## Parameter dan hasil

| Model | Arm dan scenario | LR | Batch | Epoch final | Mean CV ± SD fold | Test F1 seed 42 | Mean test ± SD lima seed |
|---|---|---:|---:|---:|---:|---:|---:|
| Encoder | window_patient S09 | 1e-3 | 64 | 5 | 0,6606 ± 0,1225 | 0,5053 | 0,5580 ± 0,0546 |
| CNN patch transformer | window_patient S06 | 5e-4 | 64 | 17 | 0,6590 ± 0,0569 | 0,6588 | 0,5747 ± 0,0548 |
| **Hierarchical — primer** | **window_segment S04** | **5e-4** | **16** | **9** | **0,6624 ± 0,0583** | **0,5837** | **0,5783 ± 0,0783** |
| ExtraTrees | window_patient T09 | — | — | — | 0,6361 ± 0,0378 | 0,5986 | 0,6053 ± 0,0186 |

ExtraTrees: 500 pohon, max_depth 12, max_features 1,0, min_samples_leaf 4. Semua pemenang memakai **window 2,5 detik**, bukan kandidat 0,5 detik. Epoch neural merupakan median best epoch lima fold konfigurasi terpilih. Lima fold hanya pada 230 development; test 59 pasien dievaluasi tanpa k-fold.

CV Hierarchical unggul hanya 0,0018 dari Encoder dan 0,0033 dari Patch. Ini memenuhi aturan pemilihan yang telah ditetapkan, bukan bukti bahwa arsitektur Hierarchical secara umum lebih unggul. CV dipakai memilih parameter dan early stopping; skor CV terpilih memiliki optimisme seleksi dan bukan nested CV.

![CV dan test](../../results/research_v3/20261001_patient_r_candidates_v3/analysis_20261002_complete/selected_cv_and_test.png)

## Apa yang diperiksa

- 630 kasus mencakup empat arm × tiga neural model × sembilan scenario × lima fold, serta 90 CV ExtraTrees. Identitas scenario/seed/fold, train IDs, probabilitas, prediksi, best epoch/history, pooled OOF dan tabel ringkasan konsisten.
- Manifest validation dan provenance sepuluh scaler fold sesuai dengan training masing-masing fold. Raw development, 230 shard kandidat dan array gabungan cocok dengan hash/konten yang dicatat; kohort/label tetap.
- Pemilihan setiap model, model primer dan epoch final direkonstruksi dari hasil CV. Config/case hash 20 final sesuai; semua checkpoint dapat dimuat, parameter neural finite dan bentuk state dict cocok. Scaler final hanya development.
- Setiap prediksi test mencakup 59 pasien unik yang sama, label/support 14 LVOT, 21 RVOT cusp, 24 RVOT septal/free-wall/other. Enam metrik, confusion matrix, laporan tiap kelas dan ringkasan tersimpan cocok.

Audit konsistensi tidak berarti seluruh sumber kebocoran klinis terbukti tidak ada. Audit ini tidak menjalankan ulang inference dari checkpoint untuk mereproduksi setiap probabilitas, tidak melatih ulang model dan tidak memvalidasi diagnosis ECG.

## Temuan yang menentukan perbaikan

### 1. Kandidat pendek tidak memberi manfaat pada setup ini

Efek di bawah membandingkan **scenario/fold yang sama**. Delta adalah kandidat_patient minus window_patient; masing-masing 45 pasangan.

| Model | Mean delta macro-F1 | Pasangan naik | Pasangan turun |
|---|---:|---:|---:|
| Encoder | −0,1051 | 1 | 44 |
| CNN patch transformer | −0,0888 | 7 | 38 |
| Hierarchical | −0,0886 | 2 | 43 |
| ExtraTrees | −0,0585 | 0 | 45 |

Dengan sampler segmen, kandidat juga lebih rendah secara mean pada ketiga neural model (−0,0739; −0,0801; −0,0854). Kandidat terbaik tidak memenangkan seleksi model manapun.

Ini hasil ablation yang perlu dilaporkan. Eksperimen sekaligus mengubah pusat crop, durasi/konteks 2,5 menjadi 0,5 detik serta jumlah/distribusi crop. Penurunan tidak mengisolasi ketepatan detector atau membuktikan label salah. R/QRS candidate belum merupakan PVC terverifikasi. Jangan membuang arm/pasien atau mengubah threshold menggunakan hasil historical test untuk menyembunyikan penurunan.

![Grid neural lengkap](../../results/research_v3/20261001_patient_r_candidates_v3/analysis_20261002_complete/full_cv_grid.png)

### 2. Efek sampling pasien berbeda dari kesimpulan sementara pilot

Pada window, patient minus segment sampler pada 45 pasangan adalah:

| Model | Mean delta | Naik/turun/seri |
|---|---:|---:|
| Encoder | +0,0325 | 32/13/0 |
| CNN patch transformer | +0,0254 | 25/20/0 |
| Hierarchical | −0,0075 | 22/22/1 |

Dengan grid penuh, sampling pasien memberi kenaikan CV rata-rata pada Encoder/Patch. Ini memperbarui kesimpulan pilot S01 yang belum konsisten, tetapi tidak berarti semua scenario atau hasil test membaik. Fold beririsan pada training; 45 pasangan bukan 45 cohort independen untuk uji p-value sederhana.

### 3. Ketidakstabilan prediksi lebih penting daripada satu skor tertinggi

| Model | Pasien dengan prediksi berbeda antar-lima seed | Proporsi |
|---|---:|---:|
| Encoder | 48/59 | 81,4% |
| CNN patch transformer | 44/59 | 74,6% |
| Hierarchical | 45/59 | 76,3% |
| ExtraTrees | 8/59 | 13,6% |

Encoder seed 42 memprediksi 13 LVOT, 46 RVOT cusp dan **0 RVOT septal/free-wall/other**. Recall kelompok terakhir 0/24. Seed 44 memprediksi 14 LVOT, **0 RVOT cusp**, 45 kelompok terakhir; recall cusp 0/21. Pada selected OOF Encoder, recall tetap 47/57 LVOT, 72/103 cusp dan 37/70 other: degenerasi pada final tertentu tidak terlihat sebagai kegagalan seluruh kelas di pooled OOF.

CV Encoder S09 memiliki SD 0,1225; best epoch fold [5,1,19,16,4]. Final 5 epoch mengikuti protokol median, bukan bug pemilihan epoch. Variasi ini menuntut evaluasi stabilitas pada development dan pemeriksaan recipe training; tidak membenarkan memilih seed test 46 atau menambah epoch menurut skor test.

Patch seed 42 F1 0,6588, tetapi mean lima seed 0,5747. Hierarchical seed 45 F1 0,4426, jauh di bawah seed lain. ExtraTrees memiliki mean tertinggi serta SD paling kecil pada lima seed test ini. Hasil belum menunjukkan keunggulan neural yang kuat terhadap comparator sederhana.

### 4. Dua subkelompok RVOT tetap sulit dibedakan

Model primer Hierarchical seed 42 benar pada 13/14 LVOT (92,9%), 12/21 RVOT cusp (57,1%), dan 9/24 RVOT septal/free-wall/other (37,5%). Confusion matrix, baris label sebenarnya dan kolom prediksi:

```text
                      LVOT  RVOT cusp  RVOT other
LVOT                    13          1           0
RVOT cusp                2         12           7
RVOT other               6          9           9
```

Ada 25 pasien development yang salah pada ketiga selected neural OOF: 5 LVOT, 8 cusp, 12 other. Ini antrean audit/review, bukan diagnosis bahwa metadata salah dan bukan alasan eksklusi. Daftar V3 dapat berbeda dari daftar V2 meskipun sama-sama berjumlah 25.

![Confusion matrix](../../results/research_v3/20261001_patient_r_candidates_v3/analysis_20261002_complete/primary_confusion_all_models.png)

## Ketidakpastian dan perbandingan V2

Bootstrap stratified 5.000 draw pasien, seed 20261002, mempertahankan jumlah tiap kelas dan checkpoint primer seed 42. CI F1 Hierarchical **0,4643–0,6973**. CI ini bersyarat pada model yang telah dilatih dan kohort historis; tidak mencakup seluruh ketidakpastian pemilihan hyperparameter/training.

Perbandingan berpasangan pada pasien yang sama, eksploratif setelah hasil test:

| Pembanding minus Hierarchical, seed 42 | Delta F1 | 95% CI pasien berpasangan |
|---|---:|---:|
| Encoder | −0,0784 | −0,1903–0,0384 |
| CNN patch transformer | +0,0751 | −0,0653–0,2153 |
| ExtraTrees | +0,0149 | −0,1168–0,1445 |

Seluruh interval mencakup nol. Ranking skor sampel ini belum memberi bukti kuat superioritas salah satu model. Model primer tetap pilihan CV.

| Model | V2 test F1 seed 42 → V3 | Mean lima seed V2 → V3 |
|---|---:|---:|
| Encoder | 0,6351 → 0,5053 | 0,6158 → 0,5580 |
| CNN patch transformer | 0,5664 → 0,6588 | 0,5876 → 0,5747 |
| Hierarchical | 0,5455 → 0,5837 | 0,5720 → 0,5783 |

**Belum ada peningkatan menyeluruh yang stabil terhadap V2.** Patch membaik pada seed 42, tetapi mean seed turun; Hierarchical mean naik sedikit; Encoder turun. Perbandingan ini tidak mengisolasi satu komponen karena V3 mengubah bobot frekuensi pasien, replacement sampling, grid terpilih dan tail patch Patch. Historical test sudah pernah diamati; ia tidak menjadi cohort baru dengan membuat run-id baru.

## SHAP primer V3 yang sudah dijalankan

Hierarchical seed 42 dijelaskan memakai expected gradients, CPU, **30 pasien background development**, **15 pasien historical test (5 per kelas)**, nsamples 100. Satu segmen tengah sepanjang 5.000 sampel atau 2,5 detik per pasien digunakan. Target adalah **logit kelas sebenarnya pada segmen tersebut**. Input berupa waveform HPF+DWT yang distandardisasi memakai scaler development final. Notebook 16 tetap menjelaskan model primer V2 yang berbeda.

Audit `analysis_v3.audit_shap` memastikan delapan hash checkpoint/scaler/kode/helper/cache sesuai dengan manifest, signature sesuai folder, sampel sesuai pool/label/koordinat dan merupakan segmen tengah. Background hanya development; tidak ada pasien background yang beririsan dengan 15 pasien test. Jumlah background per kelas adalah 4/17/9; explain set seimbang 5/5/5, sehingga distribusi untuk rata-rata ranking bukan prevalensi test asli.

Array gabungan berukuran **(15, 12, 5000)**, seluruh nilai finite, dan cocok persis dengan 15 array individual. Ranking CSV dihitung ulang dan sesuai. Grafik lead/waktu sudah diperiksa tampilannya. Pemeriksaan ini mengaudit konsistensi artefak, tidak menghitung ulang seluruh SHAP atau mengukur akurasi interpretasi klinis.

| Peringkat pada 15 pasien | Sadapan | Mean absolute SHAP logit |
|---|---|---:|
| 1 | V2 | 0,00066722 |
| 2 | V4 | 0,00065850 |
| 3 | V1 | 0,00050323 |

Pada lima sampel tiap kelas, tiga teratas LVOT adalah V4/V2/V3; RVOT cusp V2/V1/V4; RVOT septal/free-wall/other V2/V1/V4. Ini deskripsi ketergantungan model pada sampel dan background tersebut. Ia belum membuktikan bahwa sadapan itu merupakan penanda anatomi klinis terbaik, bahwa beat tertentu adalah PVC, atau bahwa interpretasinya stabil terhadap background/nsamples. Magnitudo logit SHAP antarmodel V2/V3 tidak dapat dijadikan ukuran keunggulan langsung.

![SHAP primer V3](../../results/research_v3/20261001_patient_r_candidates_v3/explain_primary_v3/729aa202d61f/lead_importance.png)

Forward CPU pada 15 segmen terpilih digunakan untuk menambah konteks. Prediksi agregat pasien benar **9/15**: 5/5 LVOT, 2/5 cusp, 2/5 kelompok terakhir. Prediksi segmen benar juga 9/15, tetapi ada **satu pasien yang kelas prediksi segmennya berbeda dari agregasi pasien**. Angka subset ini bukan skor evaluasi baru dan tidak menggantikan hasil 59 pasien. Dua contoh RVOT dalam grafik temporal salah pada tingkat pasien maupun segmen; kurva merah tetap menjelaskan logit label sebenarnya. Grafik itu tidak otomatis menjelaskan alasan model memilih kelas prediksi yang salah, atau mean probability seluruh segmen pasien.

SHAP default sudah selesai; memperbesar ke 59 pasien atau menaikkan nsamples bukan keharusan untuk menyelesaikan run ini dan tidak meningkatkan F1 classifier. Bila penjelasan menjadi kontribusi utama publikasi, tahap tambahan adalah menguji sensitivitas background/Monte Carlo dan menjelaskan keluaran agregat pasien dengan target yang sesuai. Tahap tambahan tersebut belum dikerjakan.

Artefak asli: `explain_primary_v3/729aa202d61f/`. Audit baru: `analysis_20261002_complete/shap_v3_audit.json`, `shap_v3_patient_context.csv`, dan `shap_v3_class_lead_importance.csv`. `verification.json` lama tetap snapshot verifikasi sebelum pengguna menjalankan 18–19; status SHAP terbaru berasal dari audit baru dengan waktu pemeriksaan sendiri.

## Langkah berikutnya

1. **Benchmark INCART sudah selesai dan perlu dibahas:** 75 rekaman berasal dari 32 sumber Holter; 225 hash dan jumlah anotasi cocok. F1 micro semua beat 0,96862, 95% CI grup Holter 0,95373–0,98093; recall V 92,79%, CI 84,42–98,33%. I70 mengalami banyak FN, I05/I26 banyak FP, I59/I65/I67 rendah pada recall V. Agreement sadapan tinggi tidak otomatis akurat. Hasil belum memvalidasi asal anatomi atau PVC lokal; gunakan [laporan lengkap](ANALISIS_BENCHMARK_INCART_20261002.md) untuk Bab IV. Tidak perlu mengulang notebook 17.
2. **Perbaikan training berikutnya:** prioritaskan stabilitas antar-seed, pemisahan RVOT cusp vs septal/free-wall/other, dan baseline yang kuat. Susun protokol baru menggunakan development: repeated/nested patient CV, regularisasi/recipe training, kontrol durasi yang memisahkan efek crop pendek dari detector, atau pembelajaran pada bag pasien dengan agregasi yang ditetapkan. Ini rancangan lanjutan, belum diimplementasikan/training. Jangan memperluas grid, menentukan threshold atau memilih ensemble berdasarkan 59 historical test.
3. **Penulisan TA/publikasi:** masukkan hasil lengkap, comparator, ablation negatif, variasi seed, CI, definisi anatomi, SHAP V3 serta batas anotasi PVC. GAN pada preprocessing 2.000 Hz dan ablation GAN tetap pending bila diwajibkan objektif proposal. Kontribusi ilmiah, validasi pasien baru dengan label anatomi dan interpretasi klinis belum dapat dianggap selesai hanya karena notebook berhasil berjalan. Tidak ada jaminan skor 0,9 atau publikasi Q1.

Training 12–15, SHAP 16/19, benchmark 17 dan analisis 18 sudah selesai; tidak perlu diulang untuk melanjutkan tahap di atas. Implementasi protokol training lanjutan belum dilakukan.

Prioritas berikutnya adalah membangun bukti yang stabil dan dapat direproduksi. Hasil ini belum menunjukkan batas maksimal model, tetapi juga belum mendukung klaim bahwa satu perubahan kecil akan menghasilkan 0,9 pada tiga kelas.

Artefak: `results/research_v3/20261001_patient_r_candidates_v3/analysis_20261002_complete/`. `audit.json` memeriksa CV/cache/SHAP V2; `final_audit.json` memeriksa final/test; `shap_v3_audit.json` memeriksa SHAP V3; CSV dan PNG memuat analisis rinci. `analysis_20261001_pilot/` tetap snapshot pilot, tidak ditimpa. Paket `research_v3/`/V2/V1 serta kode explainer yang identitasnya tercatat pada SHAP tetap dibekukan. SHAP V3 model terlatih sudah dihitung pengguna untuk seluruh default 15 pasien.
