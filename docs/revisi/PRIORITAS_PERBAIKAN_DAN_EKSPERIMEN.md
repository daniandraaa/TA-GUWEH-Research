# Prioritas perbaikan setelah audit eksperimen revisi

**Prioritas sesudah V9 selesai — 6 Oktober 2026:** 30 kasus median epoch valid tetapi manfaat belum konsisten; Encoder F1 +0,0141 dengan recall other −9,52 pp, Hierarchical F1 −0,0139/recall other −7,14 pp. [Analisis lengkap](ANALISIS_HASIL_V9_20261006.md). Fokus berikut **kompatibilitas pretrained/adapter → pilot representasi/fine-tuning terkontrol → penguatan evaluasi/naskah**. Tidak mengulang 46–50 atau menambah epoch/GAN secara umum.

**Eksekusi prioritas stabilitas tersedia — 6 Oktober 2026:** **46 diagnosis/smoke → 47 pilot → 48–49 real-only per model → 50 audit**. Median tiga epoch inner per model/fold, 10 referensi V8 + 20 fit baru. Persiapan/diagnosis selesai; training manual menunggu pengguna. [Panduan V9](PANDUAN_MANUAL_V9.md). Berikutnya [audit representasi/pretrained dan fine-tuning](RISET_REPRESENTASI_DAN_FINE_TUNING_20261006.md); GAN belum diperbesar.

**Prioritas aktif — 6 Oktober 2026:** V8 selesai dan valid. GAN Hierarchical turun pada semua seed untuk mean fold-F1; Encoder menunjukkan gain agregat tetapi penurunan recall other dan keunggulan tipis terhadap kontrol real. [Analisis](ANALISIS_HASIL_V8_20261006.md). Prioritas berikut **stabilitas epoch/real-only → representasi/pretraining → augmentasi lanjutan hanya dengan dasar baru**. [Fase setelah V8](FASE_SETELAH_V8_20261006.md). Tidak memperbesar rasio GAN atau memilih seed test untuk mengejar angka.

**Prioritas implementasi V8 — 5 Oktober 2026:** uji manfaat GAN V7 terhadap dua kontrol classifier, macro-F1 pasien/recall kedua RVOT/variasi seed/OOF/biaya. Kode dan notebook **41–45** tersedia pada run baru, dimulai dari smoke/pilot **41–42**. [Panduan](PANDUAN_MANUAL_V8.md). Rasio satu window per bag K4 penuh dibekukan sebelum training; tidak membuka test untuk memilih rasio atau seed.

**Prioritas aktif — 5 Oktober 2026:** GAN V7/notebook 37–40 selesai dan **5/5 fold PASS** pada QC yang dihitung ulang. Fold 1 membaik 9/15 → 0/15 kegagalan kelas–keluarga dibanding V6, pada ambang tetap. [Audit lengkap](ANALISIS_HASIL_V7_20261005.md). Prioritas kini **uji manfaat augmentasi pada classifier K4 dengan kontrol setara → ablation/seed GAN bila menjanjikan → penguatan evaluasi/naskah**. [Fase classifier](FASE_CLASSIFIER_SETELAH_V7_20261005.md). Tidak perlu training ulang GAN; belum ada skor lokalisasi baru.

> **Prioritas aktif — 3 Oktober 2026:** V4/notebook 20–27 selesai dan lolos audit: 45 diagnosis, 450 CV, 75 hasil stability, 25 final dan 25 test. Selanjutnya **audit representasi pada development → ablation terkontrol pada run baru → GAN/evaluasi/naskah yang masih terbuka**. Dua kelompok RVOT dan variasi seed tetap masalah utama; dua model baru belum meningkatkan skor. [Analisis V4 dan daftar kasus sulit](ANALISIS_HASIL_V4_20261003.md).

> **Prioritas aktif 2 Oktober 2026:** V3/SHAP/benchmark INCART sudah selesai. Langkah pertama adalah **diagnostik/stabilitas tiga classifier pada development**, lalu GAN baru dan perbandingan baseline/tuning/GAN sesuai proposal. [Urutan, anggaran dan keluaran fase berikutnya](FASE_LANJUTAN_SETELAH_V3_20261002.md). Bagian berikut adalah audit historis V1/V2; sampling sudah dikoreksi menjadi 2.000 Hz, SHAP V2/V3 dan variasi lima seed sudah tersedia. Status terbaru mengikuti [checklist](STATUS_PENYELESAIAN_DAN_LANGKAH_BERIKUTNYA.md).

> **Status sesudah notebook 10 (1 Oktober 2026):** analisis telah dijalankan pengguna tanpa error. Lihat [pekerjaan yang sudah selesai dan urutan tugas berikutnya](STATUS_PENYELESAIAN_DAN_LANGKAH_BERIKUTNYA.md). Prioritas aktif: anotasi/kurasi PVC development, kontribusi pasien yang seimbang, SHAP final V2, serta penyelarasan naskah. Eksperimen GAN masih perlu diselesaikan jika tetap menjadi bagian objektif proposal.


> **Hasil aktual 1 Oktober 2026:** seluruh 270 CV, 15 model final, dan 15 evaluasi test V2 telah selesai. Encoder tetap model utama dari CV; test macro-F1 seed 42 = 0,635, mean lima seed = 0,616 ± 0,034. Analisis menemukan 25 pasien development salah pada ketiga model serta kebingungan antara kelompok RVOT. Lihat [analisis lengkap](ANALISIS_HASIL_V2_20261001.md) dan notebook `10_Analisis_Hasil_V2.ipynb`. Prioritas berikutnya adalah review/kurasi PVC, kontribusi training antar-pasien, dan pengujian terkontrol; GAN baru/validasi eksternal belum selesai.


> **Pembaruan 30 September 2026 — prioritas V2:** sumber dataset telah diperiksa; sampling publik adalah **2.000 Hz**. Pengguna menyetujui koreksi tiga kelompok menjadi **LVOT / RVOT cusp / RVOT septal-free-wall-other**, dengan anggota pasien yang sama. Label historis “Non-OT” (LC/AC/RC) tidak sesuai anatomi sumber. Angka lama dipertahankan sebagai hasil historis. Mulai pekerjaan baru dari notebook **06**, mengikuti [panduan V2](PANDUAN_REVISI_V2.md). Training V2 belum dijalankan; GAN baru masih tahap berikutnya.


Audit: 30 September 2026. Run `20260929_grid9_v1` dan artefaknya dipertahankan sebagai hasil yang sudah dikerjakan. Perubahan metode/training harus memakai **run ID baru**; pengguna menjalankan training secara manual setelah protokol tambahan ditetapkan bersama pembimbing. Dokumen ini membedakan masalah yang terbukti, dugaan mekanisme, dan pekerjaan penelitian berikutnya.

## Yang sudah benar dan tidak perlu diulang untuk memperbaiki angka

- Split dan CV revisi sudah berdasarkan pasien: 230 development, 59 test historis; setiap pasien development muncul sekali sebagai validation dan tidak ada ID train/test beririsan.
- Scaler dan GAN revisi hanya fit pada training fold; 135/135 kasus tuning, 15/15 baseline CV, serta enam model final/test sudah lengkap.
- Model utama dipilih dari CV development sebelum test. Angka 0,9 pada eksperimen lama adalah akurasi training, bukan hasil test. Rincian ada di [audit skor lama](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/AUDIT_SKOR_09_DAN_ARAH_PUBLIKASI.md>).

## Prioritas 1 — mutu data sintetis GAN

**Temuan terukur:** [heatmap seluruh fold](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06c_diagnostik_GAN_semua_fold.png>) membandingkan median mean perubahan absolut antar-sampel dalam skala fold yang sama. Di lima fold dan GAN final, 12 sadapan pada kelas yang diaugmentasi menunjukkan rasio GAN/real **10,3–54,7×**; pada Lead II GAN final, LVOT 23,3× dan RVOT 24,9×. [Pemeriksaan hubungan sadapan](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06d_konsistensi_sadapan.png>) menunjukkan median galat hubungan `III = II − I`, `aVR = −(I+II)/2`, `aVL = I−II/2`, dan `aVF = II−I/2` pada GAN final **29,4–87,2×** real DWT. [Data kehalusan](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06c_diagnostik_GAN_semua_fold.csv>) dan [data hubungan sadapan](</Users/daniandraaa/Desktop/TA GUWEH/results/finetune_revision/20260929_grid9_v1/visualisasi/06d_konsistensi_sadapan.csv>) tersedia untuk dihitung ulang. Hubungan sadapan mengikuti [standar ECG AHA/ACCF/HRS](https://www.jacc.org/doi/10.1016/j.jacc.2007.01.024).

**Yang perlu diperiksa:** validitas amplitudo setelah `scale=5`, kecocokan preprocessing real DWT versus sintetis, spektrum frekuensi, bentuk P/QRS/T/PVC, serta hubungan antar-sadapan pada contoh banyak pasien. Kode generator memakai empat `ConvTranspose1d`, interpolasi linear ke 5.000 sampel, dan menghasilkan 12 kanal langsung; ini **kandidat penyebab**, bukan penyebab yang sudah dibuktikan. Ukuran kehalusan dan hubungan sadapan belum menilai seluruh kualitas klinis sinyal, dan belum membuktikan GAN menyebabkan perubahan F1.

**Eksperimen koreksi yang layak diuji:** konstruksi GAN yang menghasilkan kanal independen I, II, V1–V6 lalu menurunkan III/aVR/aVL/aVF dengan persamaan sadapan; lakukan perhitungan pada skala fisik sebelum normalisasi ulang. Bandingkan juga strategi upsampling yang tidak menghasilkan artefak frekuensi tinggi. Jadikan kesesuaian distribusi kehalusan, amplitudo, hubungan sadapan, dan penilaian morfologi oleh ahli ECG sebagai pemeriksaan mutu sebelum memakai keluaran baru untuk classifier. Jangan menimpa GAN yang sudah menghasilkan run sekarang.

## Prioritas 2 — perbandingan yang mengisolasi efek GAN dan tuning

Baseline revisi saat ini tidak menggunakan GAN **dan** berbeda pada optimizer, class weighting, learning rate, serta durasi training. Karena itu selisih F1 baseline vs fine-tuning tidak mengukur efek GAN atau LR/batch secara terpisah. Rancang ablation baru pada **manifest pasien, arsitektur, evaluasi patient-level, dan anggaran tuning yang sama**:

1. Sinyal real saja, tanpa GAN dan tanpa class weighting.
2. Sinyal real dengan penanganan ketidakseimbangan yang didefinisikan dari training fold saja.
3. GAN sekarang sebagai pembanding historis yang dievaluasi di bawah resep classifier yang sama.
4. GAN yang lolos pemeriksaan mutu, bila eksperimen koreksi telah siap.

Tetapkan seluruh konfigurasi, seed, dan aturan seleksi dari development sebelum training. Gunakan mean ± SD lima fold untuk menentukan apakah tiap perubahan membantu; jangan memilih varian menurut skor 59 pasien test historis. Jika ingin menguji attention pooling/fusion dari kode lama, buat ablation arsitektur terpisah. Khusus Hierarchical, penurunan F1 test dari 0,6102 lama ke 0,5270 revisi belum dapat diatribusikan pada satu perubahan.

## Prioritas 3 — ketepatan klaim dan validasi

- **Istilah fine-tuning:** `src/experiment.py` membuat model classifier baru melalui `build_model(...)` dan melatih dari bobot awal; bobot pretrained classifier tidak dimuat. Gunakan istilah *tuning hiperparameter pada training dari awal* pada naskah, atau tambahkan eksperimen transfer learning yang benar jika itu tujuan ilmiah. Bahas perubahan istilah terhadap judul/proposal bersama pembimbing.
- **Test historis:** 59 pasien yang sama sudah dipakai pada eksperimen lama, termasuk pemilihan `best_final`. Data itu tidak kembali menjadi test yang benar-benar baru. Untuk klaim generalisasi kuat, gunakan pasien eksternal/prospektif yang belum ikut keputusan desain. Bila belum tersedia, nyatakan keterbatasannya tanpa menyebut hasil sebagai validasi eksternal.
- **Sampling rate:** konfigurasi `1.000 Hz` saat ini masih asumsi kode historis; verifikasi metadata alat/dataset sebelum menafsirkan durasi 5.000 sampel atau menyebut fitur frekuensi fisiologis.
- **Analisis kesalahan:** pada test revisi, Encoder dan PatchTST benar hanya 10/24 pasien RVOT. Analisis pengembangan per kelas harus dilakukan pada prediksi validation fold, dengan review label dan sublokasi; test dipakai untuk pelaporan final.
- **Ketidakpastian:** hasil final memakai satu seed. Untuk studi lanjutan, tentukan beberapa seed sejak awal dan laporkan variasi training serta interval kepercayaan berbasis pasien. Jangan memilih seed dengan skor test tertinggi.
- **Pelaporan ilmiah:** cocokkan lingkup naskah dengan [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8) untuk studi akurasi diagnostik atau [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) untuk pengembangan model prediksi. Skor F1 yang tinggi sendiri tidak menjamin penerimaan jurnal Q1.

Urutan praktis: **verifikasi metadata dan sinyal real → audit/koreksi GAN pada run baru → ablation yang sebanding → analisis ketidakpastian → validasi pada pasien baru → naskah**. Notebook [05_Visualisasi_Proses_dan_Hasil.ipynb](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>) memuat grafik dan tabel audit tanpa melatih ulang model.
