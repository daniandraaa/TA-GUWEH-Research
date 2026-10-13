# Status penyelesaian TA dan pekerjaan berikutnya

**7 Oktober 2026 — V12 manual selesai dan diaudit:** **62–67 lengkap tanpa error tersimpan**, **40/40 fit CV + 1 deployment** terverifikasi, 100 skor evaluasi cocok dengan prediksi. Mean fold F1 **morfologi 0,6793 ± 0,1338; morfologi+RR 0,7897 ± 0,1438**, pooled OOF RR **0,8041**. **MIT-BIH P/R/F1 0,3371/0,4912/0,3998; gate FAIL**. **68 membaca FAIL dan melewati ekspor**, sehingga belum ada kandidat lokal/input pasangan/origin model baru. Detector melewatkan **2.316/7.129 PVC**; **4.239/6.887 FP** berasal dari kandidat unmatched. Prioritas berikut ialah benchmark/perbaikan detector dan generalisasi classifier, dengan MIT-BIH diakui sebagai sumber yang telah diamati. **Tidak perlu mengulang 62–68 pada recipe yang sama.** [Analisis lengkap dan bukti audit](ANALISIS_HASIL_V12_20261007.md).

**Riwayat implementasi V12 sebelum running:** notebook **62–68** disiapkan dengan 75 rekaman/32 grup INCART, dua arm, 40 fit CV termasuk 8 pilot dan satu deployment. MIT-BIH dibuka setelah seleksi/deployment dibekukan; 68 bersyarat gate PASS. [Panduan](PANDUAN_MANUAL_V12.md) menjadi referensi run yang kini selesai. V12 tidak bergantung pada cache V10; kandidat lokal tetap pseudo-label dan tidak menentukan eksklusi pasien.

**Validasi implementasi:** 24 tes sintetis lulus, setup/sintaks ketujuh notebook valid, dan 15.070 berkas lama yang dipantau tetap identik. [Rincian implementasi, bukti dan batas interpretasi](IMPLEMENTASI_DAN_VALIDASI_V12_20261007.md).

| Tahap V12 | Implementasi | Eksekusi ECG asli |
|---|---|---|
| Persiapan dan fitur INCART, notebook 62–63 | Siap; metadata/split diaudit | Selesai 75/75, detector recall PVC 93,64% |
| Pilot/CV/OOF/deployment, notebook 64–66 | Siap; 40 fit CV + 1 deployment | Selesai; 40/40 fit dan 100 skor evaluasi terverifikasi |
| Evaluasi eksternal MIT-BIH, notebook 67 | Grup 201/202 digabung, model/threshold tetap | Selesai 48 rekaman/47 grup; gate FAIL |
| Ekspor kandidat lokal, notebook 68 | Gate, mask/fallback dan tanpa eksklusi pasien | Dilewati karena FAIL; belum menghasilkan kandidat |
| Perbaikan detector, ablation input/CNN–Transformer origin | Prioritas dari diagnosis V12/literatur tersedia | Belum diimplementasikan sebagai eksperimen berikutnya |

**7 Oktober 2026 — kajian literatur selesai:** tersedia [matriks paper, penilaian metode dan rancangan fase](KAJIAN_LITERATUR_CNN_TRANSFORMER_PVC_20261007.md), [BibTeX](REFERENSI_PVC_CNN_TRANSFORMER_20261007.bib) dan [register bukti/tingkat akses](BUKTI_PENCARIAN_PVC_20261007.json). Acuan langsung CNN–Transformer: Nakasone; ablation input: Miyamae; kontrol fitur eksternal/prospektif: Zhao. Shahsavari kini terverifikasi sebagai jurnal IEEE 2024. Prioritas ilmiah tetap identitas beat dan pemisahan kedua kelompok RVOT; empat arm fusion V11 tetap menunggu cache yang cocok. Tidak ada training/inferensi ulang, perubahan kelas/source/protokol, model final/test baru atau notebook baru dalam kajian ini. Pengguna memilih kajian saat ini, tanpa pemantauan rutin.

**7 Oktober 2026 — tiga arm fitur V11 selesai dan diaudit:** notebook **60/61 lengkap 11/11 sel tanpa error**; 150 fit, 15 pilihan C inner, dan tiga OOF berisi masing-masing 230 pasien cocok dengan hasil tersimpan. Mean fold macro-F1 **QRS/RR 0,4210 ± 0,0610; morfologi 0,5555 ± 0,1070; morfologi + RR 0,5532 ± 0,0869**. Morfologi lebih baik dari QRS/RR pada lima fold; tambahan RR hanya membaik pada dua fold. Pertukaran cusp–other dan gap training–validasi masih menjadi hambatan. **Jangan ulang 57–61 untuk tiga arm fitur.** Berikutnya pemulihan cache yang cocok dengan audit V10 untuk empat arm encoder/fusion (200 fit tertunda), serta fase identitas/agregasi beat sebelum integrasi CNN–Transformer. Belum ada final/test baru atau bukti skor 0,9 tersedia. Audit tidak melatih atau menginferensikan waveform; 3.102 berkas yang dipantau tetap utuh. [Analisis lengkap](ANALISIS_HASIL_FITUR_V11_20261007.md) · [panduan V11](PANDUAN_MANUAL_V11.md).

## Riwayat status sebelum audit tiga arm fitur

**7 Oktober 2026 — 58/59 selesai; import 60 diperbaiki:** notebook 58 menyimpan LUDB PASS, micro-F1 deteksi **0,96853**, median galat durasi 10 ms; ini benchmark pengukuran, bukan skor classifier tiga kelompok. Notebook 59 menyimpan ekstraksi **230/230 pasien**, tanpa eksklusi dan median missingness fitur 0. Notebook 60 berhenti sebelum training pada `ModuleNotFoundError: research_v11_recovery`: root project belum masuk `sys.path` kernel. Sel setup 60/61 sudah diperbaiki dan startup/cache diperiksa dari cwd notebooks dengan interpreter terisolasi. **Restart Kernel lalu 60 dari sel pertama**, run `_r1` tetap sama, hasil 58/59 dipakai kembali. Source/backend/run metadata tetap utuh (1.162 file dipantau tidak berubah). Belum ada fit classifier V11 lengkap saat diagnosis ini; empat arm encoder/fusion masih tertunda karena cache hilang, tiga arm morfologi siap untuk pilot manual. [Panduan V11](PANDUAN_MANUAL_V11.md).

**7 Oktober 2026 — perbaikan notebook 58 tersedia:** download LUDB 200/200 lengkap; belum ada benchmark waveform selesai. Penyebab error: guard disk pada percobaan pertama, lalu missing cache V10. Paket `research_v11_recovery`/runner baru memakai run `_r1`, memungkinkan LUDB dan tiga arm fitur tanpa embedding, dengan source/protokol/log awal dibekukan. Parser mempertahankan seluruh 21.965 peak beranotasi yang terbaca, mencatat 122 referensi batas tidak lengkap/tidak konsisten, tanpa menebak batas atau mengubah gate. **Restart Kernel lalu 57 → 58 dari awal**. [Diagnosis/panduan](PERBAIKAN_NOTEBOOK_58_CACHE_DAN_ANOTASI_20261007.md). 19 tes sintetis lulus; belum ada fit ECG asli. Empat arm encoder/fusion tetap tertunda sampai fitur identik audit V10 dipulihkan.

**Pembaruan aktif — 6 Oktober 2026:** notebook **56 sudah selesai 5/5 sel tanpa error**. Enam replay fit cocok dengan prediksi tersimpan; 24 pemeriksaan encoder pada tiga pasien training lulus. Gap training–validasi pada tiga refit outer fold1/seed42 sekitar 0,318–0,391 menjadi alasan menguji regularisasi dan fitur yang lebih spesifik. **V11/notebook 57–61 kini diimplementasikan**, belum dijalankan pada ECG asli: 57 persiapan/smoke sintetis → 58 benchmark onset–offset LUDB → 59 ekstraksi/QA → 60 pilot fold1 → 61 lima fold dan analisis paired. [Panduan V11](PANDUAN_MANUAL_V11.md) · [riset dan prioritas tambahan](RISET_DAN_FASE_FITUR_MORFOLOGI_V11_20261006.md). Tujuh arm, sembilan C inner per fold, **315 inner + 35 outer = 350 fit termasuk pilot**; tanpa encoder inference ulang atau final/test. Identitas beat PVC/normal masih belum tervalidasi lokal dan menjadi prioritas berikutnya. Catatan V10 di bawah mempertahankan status sebelum notebook 56 diselesaikan.

**V10 selesai dan diaudit — 6 Oktober 2026:** seluruh **51–55** lengkap tanpa error; **130/130 kasus** (99 inner, 31 outer), 130 checkpoint/scaler finite, 11 pilihan inner dan 10 pembaruan bobot tail diverifikasi. Mean fold/seed F1 **random probe 0,4415**, **pretrained probe 0,3744**; tail hanya fold1/seed42 **0,3928** versus frozen pretrained pada kasus sama **0,4454**. [Analisis lengkap](ANALISIS_HASIL_V10_20261006.md). Tidak mengulang grid/final/test. **Langkah aktif: notebook 56**, inference diagnostik dari checkpoint tersimpan dan tiga pasien inner training untuk gap training–validasi serta CPU–MPS/cache. [Panduan diagnosis](PANDUAN_DIAGNOSIS_V10_20261006.md). Kode dan hasil lama tetap utuh; belum ada training/diagnosis ECG asli baru oleh Codex.

**V9 selesai dan diaudit — 6 Oktober 2026:** notebook **46–50** lengkap tanpa error tersimpan; **20/20 fit baru + 10/10 referensi**, **308 epoch baru / 7.130 update**, 30 checkpoint dan prediksi lolos audit. Mean fold-F1 Encoder **0,5311**, Hierarchical **0,4943**; kedua recall other turun. [Analisis V9](ANALISIS_HASIL_V9_20261006.md). Tidak mengulang V9. Prioritas berikut audit pretrained/adapter dan protokol from-scratch/linear probe/fine-tuning; belum ada final/test baru.

**Riwayat persiapan V9:** notebook 46–50 telah selesai. Catatan persiapan dan median epoch dipertahankan pada [Panduan V9](PANDUAN_MANUAL_V9.md); fase aktif sekarang transfer V10.

**V8 selesai dan diaudit — 6 Oktober 2026:** **41–45 tanpa error**, **30/30 selector dan 90/90 pembanding**, 120 checkpoint finite, 2742 epoch dan 59.175 update terverifikasi. Semua metrik/OOF dan anggaran setara cocok. Encoder GAN mean fold-F1 **0,5316 vs 0,5170 real-only/0,5292 real augmentation**; Hierarchical GAN **0,4796 vs 0,5081/0,5339**. Recall other turun; manfaat belum konsisten. [Analisis lengkap](ANALISIS_HASIL_V8_20261006.md). Tidak mengulang training/final/test; [fase berikut](FASE_SETELAH_V8_20261006.md) fokus stabilitas epoch/representasi real-only; V9 kini tersedia pada pembaruan terbaru di atas.

**Implementasi V8 — 5 Oktober 2026:** protokol dan endpoint classifier terpisah tersedia; **41–42** smoke/pilot, **43–44** full Hierarchical/Encoder, **45** analisis real-patient/OOF. Tiga arm memakai jumlah epoch dari selector inner yang sama; tidak memilih checkpoint dari outer validation. **30 selector + 90 pembanding**; belum ada training/hasil classifier data asli. [Panduan V8](PANDUAN_MANUAL_V8.md).

**V7 selesai dan diaudit — 5 Oktober 2026:** notebook **37–40** lengkap tanpa error; MPS smoke serta gradien CPU→MPS lulus. Seluruh **5 GAN, 500 epoch, 960 window sintetis** dan checkpoint terakhir terverifikasi; QC native direkonstruksi dan **5/5 fold PASS**, 75/75 kombinasi kelas–keluarga lolos. Masih ada 141/3390 pemeriksaan fitur gagal; PASS tidak berarti semua fitur sesuai. [Analisis V7](ANALISIS_HASIL_V7_20261005.md) · [audit mesin](../../results/research_v7/20261004_fit_distribution_gan_v7/analysis_20261005_complete/audit.json). **Tidak perlu mengulang 37–40.** Berikutnya protokol/endpoint classifier terkontrol pada run baru, belum ada skor classifier/final/test baru. [Fase berikut](FASE_CLASSIFIER_SETELAH_V7_20261005.md).

**Hasil manual V6 — 4 Oktober 2026:** **notebook33–35 selesai**, MPS smoke lulus dan GAN fold1 selesai100epoch/192sintetis. Pilot belum lolos mutu:9/15 kombinasi kelas–keluarga gagal; copy/duplikasi/numerik dan limb lead lolos. Notebook36 berhenti sesuai prasyarat sebelum fold2–5. [Analisis error, bukti dan prioritas perbaikan](ANALISIS_PILOT_GAN_V6_20261004.md). Langkah berikut adalah revisi fidelity GAN pada run baru lalu pilotfold1; belum ada classifier/final/test baru.


**V5/notebook 29–32 selesai**, kedua smoke MPS lulus dan 30/30 training baru terverifikasi bersama 30 kontrol. K8 menurunkan mean development macro-F1 Hierarchical 0,6525 → 0,6367, Encoder 0,6534 → 0,6427; pertahankan K4. Tidak ada model final/test baru dari V5. [Analisis lengkap V5](ANALISIS_HASIL_V5_20261004.md). Perbandingan classifier GAN dengan baseline K4 menjadi fase sesudah pilot V6.

**Status V4 — 3 Oktober 2026:** **notebook 20–27 selesai tanpa error tersimpan**. Diagnosis, full grid, stability, seluruh model final dan historical test telah diaudit; enam metrik cocok dengan prediksi tersimpan dan 25 checkpoint dapat dimuat. Model primer dari CV tetap Hierarchical; CV macro-F1 0,6616 ± 0,0797, test seed 42 0,6198, mean lima seed 0,5421 ± 0,0845. [Analisis lengkap dan prioritas berikutnya](ANALISIS_HASIL_V4_20261003.md) · [panduan](PANDUAN_MANUAL_V4.md).

**Notebook 28 selesai dan terverifikasi:** **230/230 pasien development, 2.274 window dan 12 pratinjau**. Tidak ditemukan kanal numerik konstan/identik atau overlap cache bermasalah; enam window pada tiga pasien tidak mempunyai kandidat konsensus lengkap. Daftar review mempunyai 37 kasus sulit, 32 kontrol unik; lima pasangan memakai ulang kontrol. [Analisis audit input](ANALISIS_AUDIT_INPUT_V4_20261003.md). Ablation K4/K8 pada notebook 29–32 juga sudah selesai: batch pasien tetap delapan, exposure setiap pasien satu kali per epoch, 30 fit baru dan 30 kontrol terverifikasi. [Panduan V5 sebagai referensi](PANDUAN_MANUAL_V5.md) · [Urutan fase lanjut](FASE_SETELAH_V4_20261003.md).

| Fase V4 | Implementasi | Eksekusi data asli |
|---|---|---|
| Protokol/hash/cache/fold/scaler | Siap, notebook 20 | Selesai; scaler hanya fit pada training fold/development |
| Diagnosis native tiga pipeline/lima fold/tiga seed | Siap, 45 fit, notebook 21 | Selesai 45/45 |
| Pilot lima model/dua pooling | Siap, 50 kasus bagian grid, notebook 22 | Selesai 50/50, termasuk Lead–Time; menjadi bagian full grid |
| Full grid learning rate/batch × lima fold | Siap, 450 kasus termasuk pilot, notebook 23–25 | Selesai 450/450; metrik dan seleksi cocok dengan prediksi |
| Seleksi/stability/final/historical test | Siap dengan readiness gates, notebook 26 | Selesai: stability 75/75 (25 reuse CV), final 25/25, test 25/25 |
| Laporan/visualisasi proses | Siap, notebook 27 | Selesai notebook 27; tambahan audit/triage tersedia pada analisis V4 |
| Audit input development | Siap, notebook 28 | Selesai 230/230 pasien dan 12 pratinjau; analisis tambahan terverifikasi |
| Ablation K=4 versus K=8 | Paket/run V5 dan notebook 29–32 dipertahankan | Selesai: kedua smoke MPS, 30/30 fit, 30 kontrol dan audit; K4 tetap |
| GAN terkoreksi, pretraining, nested/external origin validation, XAI V4 | Belum diimplementasikan dalam V4 | Masih pekerjaan lanjutan |

Target aspiratif 0,9 mengacu macro-F1 pasien tiga kelompok; tidak dipakai untuk menghapus kasus atau memilih seed test. Q1 memerlukan kontribusi dan evaluasi yang kuat, belum dapat dijamin. **Bagian berikut adalah status hasil V3/V2 yang sudah selesai.**

**Status terbaru, 2 Oktober 2026:** notebook 12–15 selesai; **630/630 CV V3, 20/20 model final, 20/20 test**. Enam metrik seluruh CV/test cocok dengan prediksi tersimpan, selection/epoch direkonstruksi dari CV dan checkpoint dapat dimuat. Fingerprint V2/V3 valid. Tidak ada training baru dijalankan oleh Codex. [Analisis lengkap V3](ANALISIS_HASIL_V3_20261002.md).

**Model primer V3:** Hierarchical, window_segment S04, LR 5e-4, batch 16, 9 epoch, seed 42; CV 0,6624 ± 0,0583, test F1 0,5837, mean lima seed 0,5783 ± 0,0783. Patch test seed 42 tertinggi (0,6588), tetapi tidak menggantikan pemilihan dari CV. ExtraTrees mean test 0,6053 ± 0,0186; belum ada keunggulan neural yang kuat. Ketidakstabilan antar-seed dan pemisahan dua kelompok RVOT tetap prioritas.

**Run manual terbaru:** notebook **18 selesai 7/7 sel dan 19 selesai 3/3 sel**, tanpa error tersimpan. SHAP Hierarchical V3 sudah menjelaskan seluruh default 15 pasien (5 per kelas) dengan 30 background development; hash/sampel/array/ranking lolos audit. Nilai atribusi terbesar rata-rata pada V2/V4/V1. Target logit satu segmen per pasien, belum penjelasan seluruh probabilitas pasien atau validasi klinis. [Analisis SHAP dan batasnya](ANALISIS_HASIL_V3_20261002.md#shap-primer-v3-yang-sudah-dijalankan).

**Benchmark notebook 17 sekarang selesai:** 5/5 sel tanpa error, seluruh 75 rekaman/32 sumber Holter dan 225 hash file lolos audit. Precision semua beat 95,41%, recall 98,36%, F1 micro **0,96862**; recall beat beranotasi V **92,79%**. Kegagalan besar ada pada I70, FP I05/I26 dan recall V I59/I65/I67. Ini deteksi beat, belum validasi classifier tiga kelompok atau anotasi PVC lokal. [Analisis lengkap INCART](ANALISIS_BENCHMARK_INCART_20261002.md).

**Perbaikan classifier kini tersedia dalam V4:** diagnosis stabilitas native, label/loss pasien, dan dua arsitektur tambahan; kontrol durasi crop tetap eksperimen lanjutan. Run V3, analisis/SHAP dan benchmark sudah selesai. Tidak perlu mengulang notebook 12–19. Penulisan Bab III–V juga masih perlu dituntaskan.

**Urutan konkret berikutnya:** notebook 20 → 21 → 22 → 23–25 → 26, serta visualisasi 27. Setelah perbaikan real-only, lanjutkan GAN/pretraining/evaluasi diperkuat sesuai [daftar fase](FASE_LANJUTAN_SETELAH_V3_20261002.md). Tiga arsitektur proposal dipertahankan sebagai pembanding; dua model baru adalah perluasan sesuai permintaan terbaru pengguna.

Snapshot pilot 150/630 dan perbaikan default flags tercatat pada [analisis pilot](ANALISIS_PILOT_V3_20261001.md); angka 480 CV tersisa sudah tidak berlaku. Output pilot notebook 12–15 ada di `arsip/notebooks_sebelum_fullgrid_20261001/`.

Pengguna menyatakan tidak memiliki ahli ECG. Kandidat R/QRS tidak diklaim sebagai PVC terverifikasi, quality flag tidak menentukan eksklusi, dan label anatomi tetap. Review ahli masih merupakan keterbatasan klinis, bukan prasyarat untuk menjalankan program. Bagian berikut mempertahankan hasil V2 yang telah selesai.

Diperiksa 1 Oktober 2026 setelah pengguna menjalankan notebook 10. Notebook 06–10 semuanya memiliki execution count untuk seluruh sel kode dan tidak menyimpan error. Notebook 10 selesai 7/7 sel; ia menganalisis model yang sudah ada, sehingga tidak menghasilkan training atau skor model baru.

## Riwayat V2 yang sudah selesai

| Pekerjaan | Bukti aktual | Status |
|---|---|---|
| Audit sumber, sampling dan nama kelas | 334 pasien lokal; sampling sumber 2.000 Hz; tiga kelompok LVOT/RVOT cusp/RVOT septal-free-wall-other | Selesai untuk protokol V2 |
| Preprocessing dan scaler terpisah | 230 pasien development, scaler hanya training fold; final scaler hanya development | Selesai |
| Ablation real-only dan class weighting | 2 arm × 3 model × 9 LR/batch × 5 fold = 270 kasus | Selesai |
| Pemilihan parameter dan model utama | Berdasarkan mean macro-F1 development CV, sebelum test | Selesai |
| Training final dan variasi seed | 3 model × seed 42–46 = 15 checkpoint final | Selesai |
| Evaluasi test tanpa k-fold | 59 pasien hold-out historis; 15 evaluasi | Selesai |
| Bootstrap, diagnosis kelas, variasi seed dan perbandingan pasangan | Notebook 09–10, laporan/CSV/grafik; 270 metrik CV dan 15 metrik test dihitung ulang | Selesai |
| Verifikasi identitas eksperimen | Protokol/fingerprint V2 tetap valid setelah analisis | Selesai |

Hasil utama V2 (macro-F1):

| Model | Mean CV ± SD fold | Test seed 42 | Mean test ± SD lima seed |
|---|---:|---:|---:|
| Encoder | 0,674 ± 0,027 | 0,635 | 0,616 ± 0,034 |
| PatchTST | 0,671 ± 0,079 | 0,566 | 0,588 ± 0,019 |
| Hierarchical | 0,660 ± 0,064 | 0,545 | 0,572 ± 0,062 |

Encoder tetap model utama **V2**: weighted_real, LR 1e-4, batch 16, 10 epoch. Notebook 10 tidak mengubah pemilihan ini. Model primer **V3** adalah Hierarchical sebagaimana bagian status terbaru. Lima seed dipakai mengukur variasi training; seed terbaik pada test tidak menggantikan seed utama 42. CV terpilih bukan nested CV. Test pernah digunakan pada pekerjaan terdahulu, sehingga tetap dilaporkan sebagai hold-out historis.

## Analisis yang menentukan prioritas

- Kesulitan utama berada di dua subkelompok RVOT. PatchTST benar hanya pada 4/21 RVOT cusp di test seed 42. Pada OOF development, recall kelompok ini juga rendah: 50/103 = 48,5%.
- Dari 25 pasien development yang salah pada ketiga model, 22 berasal dari kelompok RVOT (13 cusp, 9 septal/free-wall/other). Daftar tersebut merupakan antrean review, bukan bukti label salah dan bukan daftar pasien yang boleh dibuang untuk meningkatkan skor.
- Efek class weighting tidak konsisten. Pada konfigurasi/fold yang sama, 31/45 pasangan PatchTST turun. Memilih konfigurasi weighted terbaik memberi kenaikan CV kecil, tetapi belum menunjukkan bahwa class weighting adalah perbaikan utama.
- PatchTST seed 42 mencapai macro-F1 0,928 jika dua kelompok RVOT digabung menjadi klasifikasi biner. Ini audit eksploratif setelah melihat hasil tiga kelas, pada kohort terpilih yang sama; tidak menggantikan endpoint tiga kelas.
- Ada 2.274 segmen dari 230 pasien development. Setiap pasien menyumbang 1–46 segmen, median 8. Pada real-only, seluruh segmen masuk setiap epoch sehingga pasien dengan rekaman panjang lebih sering berkontribusi. Inverse-frequency class weighting mengatur bobot antar-kelas, tetapi tidak membuat kontribusi pasien dalam satu kelas menjadi sama.

| Kelompok | Pasien development | Segmen development | Median segmen/pasien |
|---|---:|---:|---:|
| LVOT | 57 | 440 | 6 |
| RVOT cusp | 103 | 985 | 8 |
| RVOT septal/free-wall/other | 70 | 849 | 9,5 |

Data rinci jumlah segmen tersedia pada `results/research_v2/20260930_anatomy_fs2000_v2/analysis_20261001/development_segments_per_patient.csv`.

## Pekerjaan berikutnya untuk memperbaiki model

### 1. Kurasi/anotasi PVC — prioritas pertama

**Belum selesai:** tidak ditemukan berkas indeks R peak/PVC atau anotasi beat pada direktori dataset/preprocessing project. Diagnosis asal aritmia tidak menetapkan bahwa setiap potongan sinyal berisi PVC. Detektor R peak saja tidak otomatis membedakan beat sinus dan PVC.

**Pekerjaan tanpa ahli yang sudah dilakukan:** kandidat otomatis dengan quality/fallback flags dan benchmark anotasi beat publik INCART. Recall V agregat 92,79%, tetapi 1.442 beat V terlewat dan recall beberapa rekaman di bawah 50%; agreement tinggi juga dapat disertai FP besar. Jangan mengganti label ablation-site, menganggap kandidat sebagai PVC, atau membuang pasien sulit. Preview engineering menggunakan 12 pasien development acak, bukan hanya pasien salah. Review klinis PVC/VT masih belum tersedia; batas ini dinyatakan dalam naskah dan tidak menghalangi eksperimen komputasional berikutnya.

**Keluaran:** anotasi indeks sampel beat yang disetujui, reviewer, sampling rate, kualitas sinyal dan alasan keputusan. Diagnosis anatomi tidak diganti hanya karena model salah.

**Engineering V3 sudah tersedia:** masukan dapat ditelusuri ke indeks candidate/fallback dan aturan detector tetap, jumlah kegagalan/QC dilaporkan, pasien tetap pada split yang sama. Benchmark INCART sudah memberi hasil deteksi beat dan analisis kegagalan. Validasi klinis PVC lokal tetap belum tersedia. Benchmark tidak memakai cap 64 crop; recall tersebut bukan proporsi V yang masuk training lokal. Test tidak dipakai menentukan preprocessing baru.

### 2. Kandidat R/QRS dan kontribusi antar-pasien — ablation V3 selesai

**Hasil:** empat arm V3 telah diuji pada seluruh grid. Window 2,5 detik memenangkan semua model; kandidat 0,5 detik lebih rendah secara mean pada pasangan scenario/fold yang sama. Patient sampling pada window memberi mean delta +0,0325 Encoder, +0,0254 Patch, −0,0075 Hierarchical. Kandidat tetap bukan PVC terverifikasi. [Analisis lengkap](ANALISIS_HASIL_V3_20261002.md).

**Keluaran tersedia:** protokol/fold tetap, 630 CV, selection dari development, 20 final/test, analisis pasangan dan variasi seed. Efek representasi juga mencakup perbedaan durasi/konteks dan jumlah crop; tidak mengisolasi ketepatan detector.

**Tindak lanjut:** stabilitas antar-seed neural belum baik. Recipe training, kontrol durasi, agregasi/bag pasien atau repeated/nested CV harus dirancang pada development dalam protokol baru. Model/seed primer tetap pilihan CV; jangan memilih ensemble atau seed terbaik dari test.

### 3. GAN pada preprocessing yang sudah dikoreksi

GAN V1 sudah dijalankan, tetapi GAN yang dikoreksi dan ablation augmentasi pada preprocessing 2.000 Hz belum selesai. V2 real-only/class-weight adalah eksperimen diagnosis; ia belum menggantikan eksperimen GAN jika GAN masih menjadi bagian objektif proposal.

**Keluaran:** generator kandidat, pemeriksaan morfologi/relasi sadapan/amplitudo/spektrum/keragaman pada training saja, dan perbandingan augmentasi dengan recipe classifier serta preprocessing yang sama. Data sintetis tidak masuk validation/test. Tetapkan desain sebelum membuka hasil test run berikutnya.

## Pekerjaan untuk menuntaskan laporan TA

### 4. SHAP model final V2 — selesai

Notebook 16 menghasilkan artefak Encoder V2 seed 42 di `results/research_v2/20260930_anatomy_fs2000_v2/explain_patient_v2/205d8101dd81/`. Background 30 pasien development; explain set 15 pasien historical test, 5 per kelas, satu segmen tengah/pasien, `nsamples=100`. Identitas checkpoint, signature, array finite dan lead importance lolos audit.

**Batas interpretasi:** target true-class segment logit, bukan keseluruhan probabilitas agregat pasien. SHAP menjelaskan ketergantungan keluaran model pada masukan; tidak membuktikan sebab fisiologis atau validitas diagnosis. Sensitivitas background/jumlah sampel masih perlu bila interpretasi menjadi klaim utama.

**SHAP primer V3 — selesai pada default:** notebook 19 menghasilkan 15 array untuk Hierarchical seed 42, agregat (15,12,5000), CSV dan dua grafik di `explain_primary_v3/729aa202d61f/`. Audit `analysis_v3.audit_shap` memeriksa hash checkpoint/scaler/code/helper/cache, sampel, koordinat, tidak adanya overlap background-test dan ranking. Seluruh nilai finite dan array individual cocok. Ini 15 pasien, belum seluruh 59; uji sensitivitas background/nsamples dan penjelasan agregasi pasien masih pending bila menjadi klaim utama.

### 5. Revisi naskah Bab III–V dan kesesuaian proposal

- **Bab III:** koreksi sampling 2.000 Hz, tiga nama kelompok anatomi, durasi segmen 2,5 detik, 289 pasien yang masuk target serta 45 yang dikecualikan (5 Summit, 40 unknown), split 230/59, grid 9 kombinasi, scaler per fold, dua arm V2, lima seed dan aturan model utama. Jelaskan bahwa classifier dilatih dari awal dan menjalani tuning hiperparameter; istilah fine-tuning/pretrained perlu disesuaikan dengan metode yang sebenarnya. Bedakan eksperimen GAN V1, diagnosis V2 dan rencana eksperimen berikutnya.
- **Bab IV:** tabel CV, hasil primer seed 42, mean/SD lima seed, CI bootstrap pasien, confusion matrix, kesalahan tiap kelas, hasil SHAP V3 primer serta V2 historis, benchmark INCART/CI grup Holter/kegagalan detector dan pembahasan keterbatasan. Pooled OOF dan mean fold F1 dilaporkan dengan nama yang berbeda. F1 deteksi beat tidak menggantikan F1 asal anatomi; analisis biner diberi label eksploratif.
- **Bab V:** simpulkan hasil tiga kelas yang benar, batas generalisasi dan pekerjaan lanjutan. Akurasi training sekitar 0,9 atau hasil biner eksploratif tidak menggantikan hasil test tiga kelas.
- **Proposal:** source naskah editable belum ditemukan pada audit ini. Terapkan perubahan ke dokumen yang digunakan bersama pembimbing, terutama definisi kelas dan objektif GAN/tuning.

### 6. Penguatan untuk target publikasi

Validasi pada pasien baru/eksternal, kontribusi ilmiah, penjelasan dasar kelompok anatomi, analisis ketidakpastian dan reproduktibilitas belum boleh dianggap selesai hanya karena notebook berhasil berjalan. Pasien test historis tidak menjadi pasien baru dengan mengganti split. Validasi eksternal merupakan penguatan penelitian, bukan syarat otomatis kelulusan TA atau jaminan terbit Q1.

## Checklist aktif 2 Oktober 2026

- [x] Notebook 06–10 selesai dan artefak V2 terverifikasi.
- [x] CV 270 kasus, 15 final, 15 test, bootstrap dan diagnosis kesalahan.
- [x] Daftar 25 pasien development untuk prioritas review tersedia.
- [ ] Anotasi/review PVC dan kualitas sinyal bersama ahli.
- [x] Persiapan 4.222 crop kandidat R/QRS dari 230 pasien development; audit raw/cache/scaler/fold.
- [x] Pilot neural S01: 60 CV; grid comparator ExtraTrees: 90 CV; analisis pasangan/recall tersedia.
- [x] Full grid V3: 540 neural + 90 ExtraTrees = 630 CV; kandidat belum merupakan PVC terverifikasi.
- [x] Seleksi, 20 training final dan 20 evaluasi historical test V3, seluruhnya terverifikasi.
- [x] Audit/analisis lengkap V3 dan notebook 18 selesai dijalankan 7/7 sel.
- [x] SHAP model primer Hierarchical V3 default 15 pasien; notebook 19 selesai 3/3 sel, artefak terverifikasi.
- [x] Benchmark detector INCART; notebook 17 selesai pada seluruh 75 rekaman, hasil/hash/anotasi/metrik lolos audit, grafik dan CI grup Holter tersedia.
- [ ] GAN terkoreksi dan ablation yang diperlukan objektif proposal.
- [x] SHAP Encoder final V2: 30 background/15 explain pasien, metadata dan array terverifikasi.
- [ ] Revisi Bab III dan source proposal.
- [ ] Pembahasan Bab IV serta kesimpulan/keterbatasan Bab V.
- [ ] Validasi pada pasien baru bila tersedia untuk penguatan publikasi.

Langkah langsung: baca [analisis INCART](ANALISIS_BENCHMARK_INCART_20261002.md), lalu susun protokol training berikutnya pada development untuk stabilitas antar-seed, pemisahan subkelompok RVOT serta kontrol durasi/crop. Implementasi/training protokol lanjutan belum dikerjakan. Notebook 12–19 saat ini sudah selesai; tidak perlu diulang. Instalasi/unduh pada notebook 17 boleh dinonaktifkan jika sengaja menjalankan ulang benchmark yang sama.

## Kode/perlengkapan yang sudah disiapkan pada revisi ini

- [x] Audit 20 referensi, koreksi tabel literatur dan BibTeX inti terverifikasi.
- [x] Addendum metodologi tanpa ahli ECG baru; label anatomi tetap.
- [x] Empat arm V3, kontrol jumlah update, sampling/class weights pasien, cache kandidat/fallback, manifest/fingerprint dan resume.
- [x] Koreksi sisa patch V3 dan baseline ExtraTrees.
- [x] Endpoint SHAP V2 dengan background satu segmen/pasien development dan resume per pasien.
- [x] Benchmark annotated INCART sebagai endpoint manual, tanpa perubahan package training otomatis.
- [x] Notebook manual 11–19, visualisasi QC, hasil CV/test dan SHAP primer V3 disiapkan.
- [x] Training nyata V3 lengkap: 630 CV, 20 final dan 20 test; metrik cocok dengan prediksi tersimpan.
- [x] Checkpoint final dapat dimuat, selection/epoch direkonstruksi dari CV, bootstrap pasien dan variasi seed tersedia.
- [x] SHAP V2 default penuh: 15 pasien dijelaskan; belum mencakup seluruh 59 pasien test.
- [x] SHAP V3 pada model final terpilih: seluruh default 15 pasien; belum seluruh 59 pasien test.
- [x] Benchmark nyata INCART: 75 hasil di `848d6db41c5d/`; audit 225 hash/75 anotasi, CI 32 sumber Holter, grafik kegagalan dan empat pengujian analisis lolos.
- [ ] Anotasi PVC klinis lokal, GAN terkoreksi, nested/repeated CV dan validasi eksternal asal anatomi: masih pending.

Laporan rinci: [Analisis hasil V2](ANALISIS_HASIL_V2_20261001.md). Dasar penggunaan jendela PVC terdapat pada [paper sumber](https://pmc.ncbi.nlm.nih.gov/articles/PMC8962834/).
