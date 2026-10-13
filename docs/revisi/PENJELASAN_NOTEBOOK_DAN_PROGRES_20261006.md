# Penjelasan notebook dan progres penelitian ECG

Per 6 Oktober 2026, rangkaian notebook aktif 00–56 mencakup revisi tuning, koreksi anatomi dan sampling, pembandingan arsitektur, perbaikan GAN, eksperimen augmentasi, aturan epoch, serta transfer learning ECG-FM. Notebook 51–56 sudah selesai. V10 menghasilkan 130 kasus training terverifikasi, tetapi pretrained belum memberi peningkatan yang konsisten. Diagnosis notebook 56 menunjukkan gap training–validasi besar pada tiga model outer fold 1; reproduksi checkpoint dan pemeriksaan encoder CPU–MPS/cache pada tiga pasien lolos.

Penjelasan ini membedakan hasil eksperimen yang telah selesai, diagnosis penyebab yang masih berupa hipotesis, dan pekerjaan berikutnya. Seluruh notebook aktif telah diperiksa execution count dan error tersimpannya. Notebook 36 merupakan pengecualian: ia berhenti pada prasyarat mutu GAN V6, sesuai protokol; perbaikannya diselesaikan dalam V7. Notebook dan hasil eksperimen lama dipertahankan.

**Cara membaca hasil**

- **Macro-F1 pasien tiga kelas** adalah ukuran utama: F1 dihitung untuk setiap kelas, kemudian dirata-ratakan dengan bobot sama. Nilai ini berbeda dari accuracy, recall, AUC, maupun skor training.
- **Development** adalah 230 pasien yang dipakai untuk pengembangan metode. Seluruh window pasien tetap berada pada partisi pasien yang sama.
- **Inner validation** memilih parameter dan epoch pada V8–V10. **Outer validation** menilai model yang dilatih ulang pada outer training setelah pilihan inner ditetapkan. Outer validation tetap bagian development, bukan test eksternal.
- **OOF** menggabungkan prediksi pasien ketika pasien tersebut berada di outer validation. Satu set OOF mencakup 230 pasien; tiga seed tidak menambah jumlah pasien independen.
- **Historical test** terdiri dari 59 pasien yang telah dievaluasi pada fase lama. Penggunaan kembali pasien tersebut tidak menghasilkan test baru yang belum pernah diamati.
- **Seed** mengubah inisialisasi dan proses acak training. Lima atau tiga seed menghasilkan beberapa checkpoint untuk mengukur variasi, bukan otomatis sebuah ensemble.
- Hasil CV terpilih V1–V4 dan stabilitas V5 memakai aturan seleksi yang berbeda dari fixed refit V8–V10. Angkanya tidak boleh dibandingkan sebagai efek kausal satu komponen.

**Notebook 00–05: revisi tuning awal**

| Notebook | Proses yang dilakukan | Hasil dan maknanya |
|---|---|---|
| 00 Mulai di Sini | Persiapan data development, lima fold, scaler dan GAN per training fold. | Persiapan bersama tiga arsitektur tersedia. Audit berikutnya menemukan mutu sinyal GAN perlu diperbaiki. |
| 01 Tuning Encoder | Menguji 9 pasangan learning rate–batch pada 5 fold: 45 kasus. | Konfigurasi terpilih S09, LR 1e-3, batch 64; CV macro-F1 0,6806 ± 0,0439. |
| 02 Tuning PatchTST | Grid yang sama untuk CNN-PatchTST: 45 kasus. | S02, LR 1e-4, batch 32; CV 0,6586 ± 0,0491. |
| 03 Tuning Hierarchical | Grid yang sama untuk CNN-Hierarchical Transformer: 45 kasus. | S06, LR 5e-4, batch 64; CV 0,6758 ± 0,0439. |
| 04 Final dan Evaluasi | Seleksi dari 135 CV, 15 pembanding baseline CV, pelatihan 3 model hasil tuning dan 3 baseline, evaluasi historical test tanpa k-fold test, SHAP dan laporan. | Encoder menjadi model primer dari CV. F1 test hasil tuning: Encoder 0,6367, PatchTST 0,6775, Hierarchical 0,5270. Test tertinggi PatchTST tidak mengganti model primer setelah hasil test diketahui. |
| 05 Visualisasi Proses dan Hasil | Grafik alur, fold, GAN, grid, kurva training, metrik pasien dan SHAP. | Menunjukkan masalah mutu GAN. Rasio roughness sintetis/real sekitar 10,3–54,7 kali pada audit lima fold dan GAN final; menjadi dasar audit dan perbaikan berikutnya. |

Istilah fine-tuning pada fase awal merujuk tuning recipe model yang dilatih dari bobot awal. Transfer learning dari checkpoint pretrained baru diuji secara nyata pada V10. [Analisis revisi awal](ANALISIS_HASIL_EKSPERIMEN.md).

**Notebook 06–10: V2 koreksi sumber dan anatomi**

| Notebook | Proses yang dilakukan | Hasil dan maknanya |
|---|---|---|
| 06 Audit Sumber dan Preprocessing | Memeriksa sumber, sampling 2.000 Hz, anggota kelompok dan preprocessing. | Nama kelompok diperbaiki menjadi LVOT, RVOT cusp, RVOT septal/freewall/other. Kelompok LC/AC/RC tidak lagi disebut Non-OT. Pasien dan endpoint tiga kelas dipertahankan. |
| 07 Ablasi Real Only | Tiga arsitektur × 9 skenario × 5 fold tanpa GAN: 135 kasus. | Kontrol real-only tersedia. CV terbaik per model: Encoder 0,6655, PatchTST 0,6631, Hierarchical 0,6598. |
| 08 Ablasi Class Weight | Grid yang sama dengan inverse frequency segmen training: 135 kasus. | CV terbaik weighted: Encoder 0,6738, PatchTST 0,6712, Hierarchical 0,6515. Efek weighting tidak konsisten pada semua pasangan konfigurasi/fold. |
| 09 Seleksi Final dan Laporan | Memilih arm dan parameter dari 270 CV; melatih 3 model × 5 seed dan mengevaluasi 15 historical test. | Encoder weighted menjadi primer. F1 test seed 42: Encoder 0,6351; PatchTST 0,5664; Hierarchical 0,5455. |
| 10 Analisis Hasil | Rekonstruksi metrik, variasi seed, kesalahan pasien, ketidakpastian dan pemeriksaan biner eksploratif. | Kesulitan utama pemisahan dua kelompok RVOT. PatchTST mencapai F1 biner 0,928 setelah dua RVOT digabung; angka ini tidak menggantikan hasil tiga kelas. |

Mean F1 test lima seed V2: Encoder 0,6158 ± 0,0344; PatchTST 0,5876 ± 0,0193; Hierarchical 0,5720 ± 0,0623. [Analisis V2](ANALISIS_HASIL_V2_20261001.md).

**Notebook 11–19: V3 pipeline pasien, kandidat QRS dan penjelasan model**

| Notebook | Proses yang dilakukan | Hasil dan maknanya |
|---|---|---|
| 11 Audit Referensi dan Persiapan | Audit rujukan, persiapan development dan beberapa arm window/kandidat QRS. | Pipeline V3 tersedia; kandidat R/QRS otomatis tetap belum merupakan PVC terverifikasi. |
| 12 CV Encoder | 4 arm × 9 skenario × 5 fold: 180 kasus. | Pilihan window_patient S09; CV 0,6606 ± 0,1225. |
| 13 CV Patch Transformer | Grid 180 kasus pada CNN patch transformer. | Pilihan window_patient S06; CV 0,6590 ± 0,0569. |
| 14 CV Hierarchical | Grid 180 kasus pada Hierarchical. | Pilihan window_segment S04; CV 0,6624 ± 0,0583; menjadi primer V3. |
| 15 ExtraTrees Seleksi Final dan Laporan | 90 kasus baseline fitur deskriptif; seleksi sesudah total 630 CV; 4 model × 5 seed final/test. | F1 test seed 42: Encoder 0,5053; Patch 0,6588; Hierarchical 0,5837; ExtraTrees 0,5986. Belum ada keunggulan neural yang stabil. |
| 16 SHAP Model Final V2 | Expected gradients pada model primer V2, 30 pasien background, 15 pasien penjelasan. | Atribusi logit kelas sebenarnya untuk satu segmen/pasien tersedia; bukan penjelasan seluruh probabilitas pasien atau validasi klinis. |
| 17 Benchmark Detektor INCART | Menilai detektor kandidat terhadap anotasi beat pada 75 rekaman INCART. | F1 deteksi beat 0,96862; recall beat beranotasi V 92,79%. Ini tugas deteksi beat, bukan lokalisasi tiga kelas. |
| 18 Analisis Hasil Final V3 | Audit 630 CV, 20 final/test, ketidakpastian dan variasi seed. | Mean F1 test Hierarchical 0,5783 ± 0,0783; terdapat ketidakstabilan prediksi dan kehilangan kelas pada beberapa final Encoder. |
| 19 SHAP Model Primer V3 | Menjelaskan Hierarchical seed 42 pada 15 pasien dengan 30 background development. | Sadapan V2, V4 dan V1 mempunyai atribusi absolut rata-rata tertinggi pada sampel ini. Makna fisiologisnya belum divalidasi ahli ECG. |

[Analisis V3](ANALISIS_HASIL_V3_20261002.md) dan [benchmark INCART](ANALISIS_BENCHMARK_INCART_20261002.md).

**Notebook 20–28: V4 training bag pasien dan tambahan arsitektur**

| Notebook | Proses yang dilakukan | Hasil dan maknanya |
|---|---|---|
| 20 Persiapan Protokol | Input native 2.000 Hz/2,5 detik, fold pasien dan scaler training. | Pipeline V4 tersedia, memakai satu label untuk bag pasien. |
| 21 Diagnosis Stabilitas Native V3 | Tiga pipeline terpilih × 5 fold × 3 seed: 45 fit. | Log gradien, BatchNorm, paparan sampler, recall dan jumlah prediksi kelas untuk diagnosis stabilitas tersedia. |
| 22 Pilot Model dan Pooling | 5 model × mean/attention pooling × 5 fold: 50 kasus. | Pilot seluruh model selesai, termasuk dalam full grid. Menilai proses/memori; bukan keputusan final dari satu pilot. |
| 23 Grid Encoder Patch Hierarchical | 3 model, masing-masing 90 kasus: 2 pooling × 9 parameter × 5 fold. | CV terpilih Encoder 0,6596; PatchTST 0,6448; Hierarchical 0,6616. |
| 24 Grid Residual Multiscale | CNN residual multiskala + Transformer, grid 90 kasus. | CV terpilih 0,6269 ± 0,0766; belum mengungguli tiga pembanding. |
| 25 Grid Lead Time | Pemodelan temporal per sadapan dan hubungan antar-sadapan, grid 90 kasus. | CV terpilih 0,5996 ± 0,0544; belum mengungguli pembanding. |
| 26 Seleksi Stabilitas Final Test | 450 grid, 75 hasil stabilitas termasuk 25 kasus reused, 25 final dan 25 historical test. | Hierarchical primer: test seed 42 F1 0,6198, tetapi mean lima seed 0,5421 ± 0,0845. Mean test Encoder 0,5942; PatchTST 0,6088; Residual 0,5068; Lead Time 0,4992. |
| 27 Visualisasi Proses dan Hasil | Grafik grid, training, stabilitas, OOF dan historical test. | Menunjukkan bahwa kenaikan satu seed tidak berarti kenaikan konsisten semua seed. |
| 28 Audit Input dan Window ECG | Audit 230 pasien development, termasuk 37 yang salah pada sedikitnya empat model; statistik input dan kandidat QRS. | Daftar kasus sulit dan bukti pemeriksaan teknis tersedia. Label/pasien tidak diganti atau dihapus; kandidat otomatis belum membuktikan isi PVC setiap window. |

[Analisis V4](ANALISIS_HASIL_V4_20261003.md) dan [audit input](ANALISIS_AUDIT_INPUT_V4_20261003.md).

**Notebook 29–32: V5 jumlah window training**

| Notebook | Proses yang dilakukan | Hasil dan maknanya |
|---|---|---|
| 29 Persiapan Ablation Window | Membandingkan K4/K8 dengan batch tetap 8 pasien, kontrol hasil V4. | Hipotesis menambah window diuji dengan pembanding fold/seed yang sama. |
| 30 Ablation Hierarchical | 15 fit K8: 5 fold × 3 seed. | Mean F1 K4 0,6525 → K8 0,6367. |
| 31 Ablation Encoder | 15 fit K8 pada Encoder. | Mean F1 K4 0,6534 → K8 0,6427. |
| 32 Analisis Ablation | Rekonstruksi metrik, recall dan biaya K4/K8. | K8 menurunkan F1 kedua model dan recall RVOT other, dengan waktu fit sekitar 1,5 kali. K4 dipertahankan. |

[Analisis V5](ANALISIS_HASIL_V5_20261004.md).

**Notebook 33–40: V6 dan V7 perbaikan GAN**

| Notebook | Proses yang dilakukan | Hasil dan maknanya |
|---|---|---|
| 33 Persiapan dan Smoke GAN V6 | GAN conditional 12 lead, skala global, split GAN-fit/referensi mutu di outer training. | Persiapan/smoke lolos. Reference QC terpisah dari pasien fitting GAN. |
| 34 Pilot GAN Fold 1 V6 | 100 epoch, sintesis 64 window per kelas, QC numerik dan memorisasi. | Training selesai; 192 window tersedia; 9/15 kombinasi kelas–keluarga mutu gagal. |
| 35 Visualisasi Mutu V6 | Memeriksa amplitudo, roughness, korelasi lead, spektrum, loss dan jarak ke training. | Kegagalan distribusi terlihat dan menjadi dasar revisi objective. |
| 36 GAN Fold 2–5 V6 | Gate mengharuskan fold 1 PASS sebelum melatih fold berikutnya. | Berhenti pada prasyarat, 2/7 sel dieksekusi dan RuntimeError mutu tersimpan. Fold 2–5 V6 tidak dilatih. Status ini dipertahankan sebagai hasil negatif. |
| 37 Persiapan Objective V7 | Mengganti pencocokan moment minibatch dengan target quantile distribusi amplitudo, roughness, korelasi lead dan spektrum. | Target hanya dari pasien GAN-fit; arsitektur, anggaran dan ambang QC tetap. |
| 38 Pilot GAN Fold 1 V7 | Training baru dari inisialisasi awal, 100 epoch; sintesis/QC. | Fold 1 PASS; 192 window sintetis tersedia. |
| 39 Perbandingan Mutu V6 V7 | Membandingkan mutu dengan fold, seed dan anggaran setara. | Keluarga gagal fold 1 turun 9/15 → 0/15. Ini peningkatan QC engineering, belum manfaat classifier. |
| 40 GAN Fold 2–5 V7 | Menyelesaikan fold berikutnya sesudah pilot PASS. | Seluruh 5 fold PASS, total 500 epoch dan 960 window sintetis. Sintetis tidak dihitung sebagai pasien baru. |

[Diagnosis V6](ANALISIS_PILOT_GAN_V6_20261004.md) dan [analisis V7](ANALISIS_HASIL_V7_20261005.md).

**Notebook 41–45: V8 manfaat augmentasi terhadap classifier**

| Notebook | Proses yang dilakukan | Hasil dan maknanya |
|---|---|---|
| 41 Persiapan Classifier | Dua model, tiga arm: real-only, augmentasi real, GAN; K4, batch 8 pasien dan smoke. | Protokol pembanding tersedia. Epoch dipilih inner; outer tidak dipakai early stopping saat refit. |
| 42 Pilot Fold 1 | Dua selector inner dan enam refit outer untuk dua model/tiga arm. | Delapan fit selesai, termasuk dalam 120 total fit V8. |
| 43 Hierarchical Lima Fold | 15 selector inner + 45 pembanding outer. | Mean F1 real-only 0,5081; augmentasi real 0,5339; GAN 0,4796. |
| 44 Encoder Lima Fold | 15 selector inner + 45 pembanding outer. | Mean F1 real-only 0,5170; augmentasi real 0,5292; GAN 0,5316. |
| 45 Analisis Manfaat GAN | Audit 30 selector/90 outer, metrik pasien, recall, variasi seed dan biaya. | Gain GAN Encoder kecil terhadap augmentasi real dan recall other turun; Hierarchical GAN turun pada seluruh mean seed. QC GAN PASS tidak otomatis memperbaiki lokalisasi. |

[Analisis V8](ANALISIS_HASIL_V8_20261006.md).

**Notebook 46–50: V9 stabilisasi aturan epoch**

| Notebook | Proses yang dilakukan | Hasil dan maknanya |
|---|---|---|
| 46 Diagnosis Epoch dan Persiapan | Median tiga epoch terpilih inner per model/fold, mempertahankan recipe real-only. | 10 hasil identik V8 menjadi referensi; 20 fit baru ditetapkan. Tidak mengubah split inner pasien. |
| 47 Pilot Fold 1 | Hierarchical 18 epoch dan Encoder 10 epoch, seed 42. | Dua fit baru selesai, termasuk dalam anggaran 20. |
| 48 Hierarchical Lima Fold | 15 kasus: 5 referensi lama + 10 fit baru. | Mean F1 V8 0,5081 → V9 0,4943 ± 0,1068. |
| 49 Encoder Lima Fold | 15 kasus dengan aturan median yang sama. | Mean F1 V8 0,5170 → V9 0,5311 ± 0,0814. Log sempat mencatat guard disk; run berikutnya lengkap. |
| 50 Analisis Stabilitas Epoch | Audit 30 kasus dan perbandingan aturan, recall serta OOF. | Recall RVOT other turun sekitar 7,14 poin persentase pada Hierarchical dan 9,52 poin pada Encoder. Median belum menjadi perbaikan final universal. |

[Analisis V9](ANALISIS_HASIL_V9_20261006.md).

**Notebook 51–56: V10 transfer learning dan diagnosis terbaru**

V10 menguji apakah representasi model ECG pretrained dapat membantu lokalisasi tiga kelompok pada data project. Tiga arm mempunyai peran berbeda:

| Arm | Bobot encoder | Bagian yang dilatih | Cakupan selesai |
|---|---|---|---|
| random_probe | Inisialisasi acak, lalu dibekukan | Head linear tiga kelas | 5 fold × 3 seed head |
| pretrained_probe | Checkpoint resmi ECG-FM, dibekukan | Head linear tiga kelas | 5 fold × 3 seed head |
| pretrained_tail | Checkpoint resmi ECG-FM | Layer Transformer terakhir + head; bagian sebelumnya beku | Pilot fold 1/seed 42 |

Random probe bukan kontrol encoder acak yang dilatih penuh dari awal. Ia menguji head di atas fitur acak beku. Tiga seed benchmark mengganti seed head/refit, bukan tiga pretraining encoder berbeda.

**51 Kompatibilitas dan persiapan**

Enam dari enam sel kode selesai. Notebook memverifikasi checkpoint/config resmi yang di-cache, mengecek kompatibilitas CPU yang telah disimpan, menyiapkan identitas eksperimen dan grid, kemudian menjalankan smoke forward/backward sintetis MPS. Konfigurasi mengikuti isi checkpoint: encoder 12 layer Transformer, dimensi fitur 768, sekitar 90,9 juta parameter pada model penuh. Smoke MPS PASS. Pada akhir notebook ini belum ada fit classifier ataupun skor lokalisasi.

Grid head menguji LR 1e-4, 1e-3, 1e-2 dan batch efektif 8, 16, 32 pasien. Grid tail menguji LR 1e-6, 3e-6, 1e-5 dan batch yang sama; LR head mengikuti pilihan pretrained probe fold 1. Setiap grid terdiri dari sembilan kombinasi. LR tail lebih kecil karena mengubah bobot pretrained, sedangkan head adalah layer baru. Batas head 100 epoch/patience 15; tail 30 epoch/patience 7; AdamW weight decay 1e-4 dan clipping gradien 1.

**52 Ekstraksi representasi development**

Tiga dari tiga sel selesai. Inference encoder dilakukan pada 2.274 window milik 230 pasien development, untuk random dan pretrained encoder. Input native 12 lead, 5.000 sampel pada 2.000 Hz atau 2,5 detik, diadaptasi menjadi 1.250 sampel pada 500 Hz; durasi tetap 2,5 detik. Adapter menggunakan interpolasi linear dan z-score tiap lead/window; tidak menambahkan filter antialias khusus.

Setiap window menghasilkan 78 token berdimensi 768. Rata-rata token menjadi fitur window; rata-rata seluruh window menjadi fitur pasien. Dua matriks fitur berukuran 230 × 768 disimpan; token sebelum layer terakhir pretrained juga disimpan untuk pilot tail. Cache lengkap, finite, sesuai pasien/label dan hash. Semua pasien tidak menerima vektor yang sama; audit tidak menemukan collapse numerik sederhana. Tidak ada training encoder di notebook ini.

**53 Pilot probe dan partial fine-tuning**

Lima dari lima sel selesai. Pilot memakai fold 1/seed 42: 147 pasien inner training dan 37 inner validation. Setiap arm menguji sembilan konfigurasi; konfigurasi/epoch dipilih dengan macro-F1 inner. Model kemudian diinisialisasi ulang dan dilatih pada seluruh 184 pasien outer training selama jumlah epoch yang tetap. Evaluasi hanya dilakukan sesudah refit, pada 46 pasien outer validation. Total 27 fit inner + 3 refit outer = 30 fit.

| Arm | LR terpilih | Batch pasien | Epoch refit | Inner F1 terpilih | Outer F1 | Outer accuracy |
|---|---|---:|---:|---:|---:|---:|
| Random probe | Head 1e-2 | 8 | 8 | 0,5850 | 0,4247 | 0,4348 |
| Pretrained probe | Head 1e-3 | 16 | 8 | 0,4864 | 0,4454 | 0,4565 |
| Pretrained tail | Tail 1e-6; head 1e-3 | 16 | 6 | 0,5371 | 0,3928 | 0,3913 |

Pretrained frozen membaik sedikit terhadap random pada pilot ini. Membuka layer terakhir meningkatkan inner F1, tetapi outer F1 turun 5,26 poin persentase terhadap pretrained frozen. Pilot satu fold belum mendukung klaim manfaat fine-tuning tail pada lima fold.

**54 Frozen probe lima fold dan tiga seed**

Lima dari lima sel selesai. Untuk setiap arm probe, 9 konfigurasi × 5 fold = 45 selector inner, kemudian 5 fold × 3 seed = 15 refit outer. Setiap arm memiliki 60 fit, total dua arm 120. Dua puluh kasus probe dari notebook 53 digunakan ulang sesudah verifikasi; notebook 54 menambahkan 100 fit probe. Sepuluh kasus tail dari pilot tetap dipisahkan. Total V10 menjadi 130, bukan 150.

| Arm | Mean macro-F1 15 fold/seed | SD deskriptif | Mean recall LVOT | Mean recall RVOT cusp | Mean recall RVOT other |
|---|---:|---:|---:|---:|---:|
| Random probe | 0,4415 | 0,0682 | 64,34% | 37,10% | 35,24% |
| Pretrained probe | 0,3744 | 0,0539 | 54,60% | 34,94% | 29,52% |

Pretrained turun 6,71 poin persentase dan menang pada 4/15 pasangan fold/seed. Mean masing-masing seed juga turun. Angka 15 memakai split/pasien/encoder berulang; bukan 15 pengamatan pasien independen. Kinerja pretrained belum konsisten lebih baik pada recipe V10.

Parameter head terpilih berbeda antar-fold. Tabel berikut merupakan pilihan inner masing-masing fold, belum satu konfigurasi untuk model final:

| Fold | Random: LR / batch / epoch | Pretrained: LR / batch / epoch |
|---|---|---|
| 1 | 1e-2 / 8 / 8 | 1e-3 / 16 / 8 |
| 2 | 1e-2 / 32 / 7 | 1e-2 / 32 / 12 |
| 3 | 1e-3 / 8 / 9 | 1e-2 / 32 / 27 |
| 4 | 1e-2 / 16 / 9 | 1e-3 / 16 / 1 |
| 5 | 1e-2 / 8 / 23 | 1e-3 / 16 / 9 |

[Pilihan inner terverifikasi](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/analysis_20261006_complete/choices_verified.csv). Semua seed outer pada fold yang sama memakai pilihan inner tersebut. Pilihan epoch 1 pada pretrained fold 4 merupakan maksimum inner yang tercatat; tidak diganti berdasarkan hasil outer.

**55 Analisis dan visualisasi transfer**

Enam dari enam sel selesai. Notebook merekonstruksi metrik dari prediksi tersimpan serta menampilkan grid, ringkasan, OOF, recall dan grafik. Notebook ini tidak melakukan training atau inference waveform baru. Audit tambahan pada hasil lengkap memverifikasi 130 checkpoint/scaler, 11 pilihan inner, pembaruan parameter pada 10 kasus tail, 2.945 epoch dan 33.832 optimizer update.

Masing-masing dari enam set OOF mencakup tepat 230 pasien. Mean pooled OOF random 0,4434 dan pretrained 0,3807; selisih −0,0627. Pooled OOF berbeda dari rata-rata F1 fold karena F1 nonlinear. Bootstrap pasien berpasangan/berstrata 2.000 kali menghasilkan interval 95% selisih pooled F1 [−0,1358; +0,0161]. Interval melintasi nol; hasil belum memastikan pretrained selalu lebih buruk pada populasi lain. Bootstrap ini bersyarat pada prediksi tersimpan, tidak melatih ulang model.

**56 Diagnosis tanpa training ulang**

Lima dari lima sel selesai, tanpa error tersimpan. Notebook ini menjalankan inference ulang dari checkpoint yang sudah ada, tanpa optimizer update, seleksi baru atau historical test. Dua diagnosis selesai:

1. Enam checkpoint fold 1/seed 42: inner terpilih dan outer refit dari masing-masing tiga arm. Dua belas tabel prediksi training/validation tersedia. Metrik training dihitung setelah fitting dari checkpoint yang sama dengan evaluasi, sehingga bisa dibandingkan langsung.
2. Encoder CPU dan MPS diperiksa pada tiga pasien inner training yang dipilih deterministik, satu per kelas, mencakup 30 window. Fitur pasien dan token dibandingkan dengan cache notebook 52.

| Arm, outer fold 1/seed 42 | Training macro-F1 pada 184 pasien | Validation macro-F1 pada 46 pasien | Gap training minus validation |
|---|---:|---:|---:|
| Random probe | 0,8157 | 0,4247 | 0,3910 |
| Pretrained probe | 0,7635 | 0,4454 | 0,3181 |
| Pretrained tail | 0,7703 | 0,3928 | 0,3776 |

Gap sekitar 31,8–39,1 poin persentase merupakan bukti langsung lemahnya generalisasi pada checkpoint yang diperiksa, konsisten dengan overfitting. Ini belum mengidentifikasi satu penyebab tunggal dan belum mengukur gap seluruh 130 kasus.

Untuk inner checkpoint, training/validation F1 berturut-turut: random 0,8165/0,5850; pretrained frozen 0,8380/0,4864; tail 0,7985/0,5371. Pola gap juga sudah muncul pada inner validation.

Seluruh enam prediksi validation direproduksi, argmax sama, selisih probabilitas maksimum sekitar 1,11e-16. Seluruh 24 perbandingan encoder/cache lolos toleransi tetap rtol 1e-3 dan atol 1e-4. Encoding MPS terhadap cache tepat sama pada pasien anchor; selisih CPU–MPS terbesar 8,39e-5. Pemeriksaan ini mendukung konsistensi backend/cache pada tiga pasien tersebut, bukan validasi seluruh pasien atau kesesuaian fisiologis representasi.

![Macro F1 training dan validasi pada checkpoint fold 1](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/analysis_20261006_diagnostic_review/gap_training_validation_fold1.png)

[Audit hasil lengkap V10](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/analysis_20261006_complete/audit.json), [tabel diagnosis checkpoint](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/diagnostics_20261006_v1/fold1_saved_fit_replay_mps.csv), [tabel diagnosis encoder](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/diagnostics_20261006_v1/encoder_cache_replay_mps.csv), dan [audit ulang diagnosis](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/analysis_20261006_diagnostic_review/audit.json).

**Progres yang sudah dicapai**

- Struktur eksekusi telah dipisah per tahap dan model, dengan log, resume dan artefak yang dapat diaudit; training nyata tetap dijalankan manual oleh pengguna.
- Sampling dan nama kelompok telah dikoreksi; split, scaler dan label dapat ditelusuri pada level pasien.
- Tiga arsitektur proposal, dua perluasan CNN–Transformer dan ExtraTrees telah diuji pada fase masing-masing.
- Stabilitas seed, jumlah window, mutu GAN, manfaat GAN terhadap classifier dan aturan epoch telah diuji. Hasil negatif dipertahankan.
- Transfer learning ECG-FM sudah diimplementasikan dan dievaluasi, termasuk fine-tuning layer terakhir yang benar-benar memperbarui bobot.
- Analisis kesalahan, confusion matrix, recall tiap kelas, OOF, bootstrap dan SHAP tersedia sesuai lingkup masing-masing. INCART menyediakan benchmark deteksi beat terpisah.
- Diagnosis V10 kini memberikan arah yang lebih konkret: konsistensi hasil pada pemeriksaan ini lolos, tetapi gap generalisasi besar.

**Pekerjaan yang belum selesai dan urutan berikutnya**

1. **Uji regularisasi head dengan kontrol yang sama.** Prioritas sesudah notebook 56 adalah eksperimen baru terhadap random/pretrained features yang telah dibekukan, dengan regularisasi dipilih hanya pada inner training/validation. Contohnya head linear dengan L2 eksplisit. Pada inner training sekitar 147 pasien tersedia 768 fitur; kondisi ini memotivasi uji regularisasi, belum membuktikan bahwa jumlah fitur merupakan satu-satunya penyebab. V11 belum diimplementasikan atau dilatih.
2. **Ablation adapter dan pooling secara terpisah.** Bila regularisasi belum cukup, uji satu perubahan normalisasi/pooling/durasi pada satu eksperimen, dengan random/pretrained kontrol yang sama. Durasi kontinu harus berasal dari rekaman yang sah; menyambung window overlap tidak menciptakan rekaman 5 detik yang independen. Ketidakcocokan input masih hipotesis.
3. **Perluas fine-tuning sesudah baseline terkendali membaik.** Resep tail sekarang belum mendukung perluasan otomatis ke lima fold. Kontrol arsitektur/input dari awal yang dilatih penuh diperlukan bila ingin mengisolasi manfaat pretraining terhadap from-scratch training.
4. **Tetapkan satu model final dan rancangan evaluasi yang lebih independen.** V10 tidak mempunyai model final baru atau test baru; kedua endpoint dinonaktifkan dalam protokol. Historical test sudah pernah digunakan. Independensi pasien pretraining terhadap kohort lokal juga belum diverifikasi.
5. **Tuntaskan Bab III–V dan bukti untuk publikasi.** Pisahkan tuning dari transfer learning, tampilkan semua seed dan kelas, jelaskan hasil negatif, biaya dan ketidakpastian, serta batasi klaim SHAP/QRS/GAN yang belum dinilai ahli ECG. Evaluasi eksternal lokalisasi memerlukan label asal anatomi yang sesuai; INCART tidak menggantikannya.

Tidak ada bukti bahwa kinerja saat ini sudah maksimal. Tidak ada pula bukti patient-level macro-F1 tiga kelas 0,9 pada evaluasi yang independen. Angka 0,9 pada grafik historis merupakan accuracy training: Encoder 0,948, PatchTST 0,957, Hierarchical 0,917. F1 biner 0,928 V2 dan F1 deteksi beat INCART 0,96862 berasal dari endpoint lain. Ketiganya tidak boleh dipakai sebagai hasil lokalisasi tiga kelas. [Audit skor historis](AUDIT_SKOR_09_DAN_ARAH_PUBLIKASI.md).

Kebutuhan sekarang adalah perbaikan generalisasi yang diuji dengan kontrol jelas, kemudian model final dan evaluasi yang lebih independen. Satu angka skor tidak menentukan kesiapan publikasi Q1. Notebook 00–55 tidak perlu diulang untuk menjawab diagnosis terbaru; notebook 56 juga sudah selesai.
