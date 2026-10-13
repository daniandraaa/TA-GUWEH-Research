# Fase lanjutan setelah V3 selesai — 2 Oktober 2026

**Pembaruan 3 Oktober 2026:** V4 dan notebook 20–27 telah selesai dijalankan pengguna. Diagnosis 45/45, grid 450/450, hasil stability 75/75, final 25/25 dan test 25/25 lolos audit. Model primer Hierarchical menghasilkan test macro-F1 0,6198 pada seed 42 dan mean lima seed 0,5421 ± 0,0845. [Analisis hasil V4 dan pekerjaan berikutnya](ANALISIS_HASIL_V4_20261003.md). Pembanding V3/V4 dipertahankan; prioritas beralih ke audit development dan ablation baru, sebelum GAN/evaluasi lanjutan/naskah.

**Urutan aktif:** diagnosis native → pilot/model/MIL real-only V4 → grid/stability/final yang dibekukan → GAN terkoreksi/pretraining/ablation/evaluasi diperkuat → naskah. Default V4 membutuhkan 570 fit termasuk diagnosis, grid, tambahan stability dan final; rincian pada kajian V4. Target 0,9/Q1 belum terukur atau dijamin. **Rencana awal di bawah tetap menjadi daftar pekerjaan proposal/lanjutan**; spesifikasi eksperimen yang dijalankan sekarang mengikuti protocol V4. Batasan mempertahankan tiga struktur di rencana awal kini berlaku pada pembanding proposal, sementara dua model tambahan menjadi perluasan yang diizinkan pengguna.

## Posisi penelitian

Selesai: koreksi sampling 2.000 Hz/nama tiga kelas, split pasien/scaler fold, 630 CV V3, seleksi dari development, 20 final/test, variasi lima seed, SHAP primer 15 pasien, benchmark 75 rekaman INCART dan analisis ketidakpastian/kegagalan.

Masalah utama classifier: variasi seed besar, dua kelompok RVOT sulit dipisahkan, beberapa final Encoder tidak memprediksi salah satu kelas. Hierarchical primer test macro-F1 0,5837 dan mean lima seed 0,5783 ± 0,0783. INCART F1 deteksi beat 0,9686 tidak meningkatkan atau menggantikan hasil classifier ini. Crop kandidat 0,5 detik belum memenangkan perbandingan V3 terhadap window 2,5 detik.

Kebutuhan proposal yang masih terbuka: perbandingan tiga model baseline dengan hasil tuning di bawah preprocessing yang benar dan augmentasi GAN dalam skenario tuning. ExtraTrees adalah comparator tambahan; ia tidak menggantikan baseline tiga arsitektur hybrid. GAN V1 yang memakai preprocessing/label historis tidak menjadi bukti GAN untuk protokol terkoreksi. [Proposal pengguna](</Users/daniandraaa/Downloads/PROPOSAL DANIANDRA PRAYUDISTY ILHAM BAB 1, 2, DAN 3.pdf>), bagian tujuan dan batasan penelitian, menjadi dasar cakupan ini.

## Fase 1 — diagnosis dan stabilitas classifier (kerjakan pertama)

**Tujuan:** mendapatkan bukti pada validation development tentang kapan kelas runtuh, seberapa besar variasi training, serta komponen training yang layak diperbaiki. Mengulang seed untuk diagnosis tidak otomatis meningkatkan model.

1. Buat paket/run baru yang merekam identitas protokol, serta notebook manual diagnostik. Kode dan artefak V1–V3 dipertahankan. Notebook lanjutan tidak memanggil endpoint V3 untuk mengubah seed CV, karena seed fold V3 dibekukan oleh implementasinya.
2. Gunakan 230 development, lima fold pasien yang sama, tiga arsitektur, waveform window 2,5 detik dan parameter terpilih V3 sebagai titik awal: Encoder window_patient S09 (1e-3/64), Patch window_patient S06 (5e-4/64), Hierarchical window_segment S04 (5e-4/16). Ini pengujian tiga pipeline terpilih, bukan isolasi pengaruh arsitektur saja.
3. Tetapkan tiga seed training per fold (misalnya 42/43/44) dalam protokol baru. Total **3 model × 5 fold × 3 seed = 45 fit diagnostik**. Ini belum full grid baru dan belum final training. Fold tidak diganti untuk mencari skor tertinggi.
4. Simpan per epoch: train/validation loss, LR, gradient norm, macro-F1 pasien, recall tiap kelas, jumlah pasien yang diprediksi ke tiap kelas, best epoch, paparan pasien/kelas dari sampler dan ringkasan statistik BatchNorm. Simpan checkpoint terpilih dan prediksi OOF agar dapat diaudit.
5. Inspeksi pipeline scaler, mode train/eval, sampler, bobot kelas, scheduler dan early stopping. Adanya BatchNorm/weighting bukan bukti penyebab; data log dan perbandingan terkontrol diperlukan.
6. Bila diagnosis membenarkan perubahan recipe, bandingkan satu perubahan setiap kali pada fold/seed/LR/batch yang sama. Misalnya AdamW vs Adam dengan weight decay yang sama merupakan kandidat uji, bukan perbaikan yang sudah terbukti. Satu recipe tambahan penuh membutuhkan 45 fit tambahan. Perubahan bobot kelas/sampler menjadi ablation terpisah.

**Keluaran:** tabel stabilitas validation/OOF, confusion matrix tiap seed/fold, kurva tiap kelas, daftar kasus collapse dan rekomendasi recipe dari development. Gunakan macro-F1 rata-rata dengan aturan seleksi/tie-break yang ditetapkan sebelum run; laporkan SD dan recall per kelas. Jangan memilih seed test terbaik.

**Batas:** parameter awal sudah dipilih dari CV pada development yang sama. Hasil diagnostik ini tidak disebut estimasi generalisasi independen atau nested CV. Bila tetap terjadi collapse, laporkan; jangan menyembunyikannya dengan menghapus seed/pasien.

## Fase 2 — GAN terkoreksi dan pemeriksaan mutu

**Tujuan:** memenuhi komponen augmentasi proposal dengan eksperimen pada sinyal/label yang sudah dikoreksi.

- Siapkan GAN versi baru untuk waveform 12 sadapan pada 2.000 Hz. Generator hanya melihat pasien training fold; validation/test tetap sinyal asli. Scaler dan parameter mutu hanya berasal dari training fold.
- Mulai pilot mutu pada satu training fold sebelum melatih generator semua fold. Evaluasi per kelas: amplitudo dalam skala fisik, distribusi frekuensi/kehalusan, hubungan sadapan, keragaman dan kemiripan terhadap contoh training untuk mendeteksi duplikasi/memorisasi.
- Tetapkan pemeriksaan/aturan penggunaan sintetis sebelum melihat efeknya pada validation; catat rasio real/sintetis dan provenance pasien/fold. Jangan memakai generator V1 lama pada preprocessing baru.
- Tanpa ahli ECG, hasil pemeriksaan dinyatakan sebagai mutu komputasional. Jangan menyebut morfologi PVC atau diagnosis sintetis telah tervalidasi klinis.

**Keluaran:** artefak generator, contoh sinyal, tabel/grafik mutu dan manifest. Jika mutu tidak memadai, hasil negatif tetap dilaporkan dan sintetis belum digunakan untuk eksperimen classifier utama. GAN tidak perlu menghasilkan peningkatan agar penelitian dapat dilaporkan dengan benar.

## Fase 3 — perbandingan baseline, tuning dan GAN

**Tujuan:** menjawab tujuan utama proposal dengan perbandingan yang bisa ditafsirkan.

| Skenario | Data | Parameter classifier | Pertanyaan |
|---|---|---|---|
| A: baseline hybrid | Asli | Konfigurasi awal dibekukan sebelum run | Kinerja awal tiap arsitektur |
| B: tuning real | Asli | Grid LR × batch lengkap | Dampak pencarian parameter pada pipeline real |
| C: tuning + GAN terkoreksi | Asli + sintetis training fold | Grid yang sama dan anggaran classifier sebanding | Dampak pipeline augmentasi dan tuning |

Tetap gunakan Encoder, CNN patch transformer dan Hierarchical. Struktur dasarnya dipertahankan. Preprocessing, split pasien, definisi target, evaluasi per pasien, recipe yang tidak sedang diuji serta aturan seleksi sama. Grid tetap LR **[1e-4, 5e-4, 1e-3]** × batch **[16, 32, 64]** pada lima fold pasien.

Untuk satu seed CV, B membutuhkan **135 fit**, C **135 fit**, A **15 fit**: total **285 fit classifier**, di luar GAN dan diagnosis fase 1. Tambahan seed atau sampler arm meningkatkan jumlah tersebut; tentukan anggaran sebelum menjalankan. Jumlah update classifier harus sebanding pada pasangan B/C dan biaya training GAN dilaporkan terpisah.

Bandingkan B/C pada scenario/fold/seed yang sama untuk memeriksa efek augmentasi. Selisih dua konfigurasi pemenang yang berbeda mencerminkan pipeline setelah seleksi, bukan efek murni GAN. Jangan memilih kondisi berdasarkan hasil 59 historical test. Bila sampler/weighting berbeda, catat sebagai ablation tambahan, bukan menyebut seluruh selisih sebagai efek tuning.

**Keluaran:** mean/SD CV, pooled OOF, recall tiap kelas, perbandingan pasangan serta pilihan hyperparameter/model dari development. ExtraTrees tetap comparator tambahan. Epoch final ditetapkan dari prosedur CV sebelum evaluasi test.

## Fase 4 — penguatan evaluasi untuk publikasi

Setelah recipe/ruang pencarian cukup stabil, rancang **nested patient CV**: pemilihan parameter berlangsung di inner CV dan penilaian di outer fold yang tidak ikut pemilihan tersebut. Jumlah outer/inner fold dan early stopping dalam inner training perlu dibekukan sesuai anggaran. Outer validation tidak dipakai memilih epoch. Semua scaler, GAN dan langkah seleksi diulang hanya dari data training yang sesuai.

Nested CV mengurangi bias dari memakai skor pencarian parameter sebagai estimasi performa; lihat [Varma & Simon, 2006](https://link.springer.com/article/10.1186/1471-2105-7-91). Ia tidak menghapus riwayat perubahan metode berdasarkan hasil yang sudah diamati, dan tidak menjadikan development atau test historis sebagai cohort eksternal baru.

Retraining final tetap satu checkpoint primer per arsitektur dari seluruh development; seed primer ditetapkan sebelumnya. Lima seed dapat dipakai untuk variasi training, bukan untuk memilih hasil test tertinggi. Historical test 59 pasien dievaluasi tanpa k-fold setelah keputusan lengkap dibekukan. Pasien eksternal/prospektif berlabel anatomi yang belum ikut desain memperkuat generalisasi bila tersedia; INCART hanya menjadi bukti deteksi beat.

## Fase 5 — naskah TA/publikasi (dapat dimulai sekarang)

- Bab III: sampling/durasi, tiga nama kelompok, kohort 289 dengan split 230/59, scaler per fold, grid, aturan seleksi, training dari awal/tuning hiperparameter, penempatan GAN, seed dan status test historis.
- Bab IV: CV/test tiap model, baseline vs tuning vs GAN bila fase 3 selesai, variasi seed, recall RVOT, comparator, hasil negatif kandidat pendek, INCART beserta kegagalan, SHAP dan batas target segment logit.
- Bab V: kesimpulan sesuai hasil serta keterbatasan anotasi PVC lokal, klinis, kohort dan interpretasi. Tidak mengganti endpoint tiga kelas untuk mengejar skor biner lebih tinggi.

Rumuskan kontribusi dari hasil yang dibuktikan. Angka training 0,9 atau detector F1 0,9686 tidak menjadi target numerik keberhasilan klasifikasi asal anatomi. Target Q1 mengarahkan kekuatan metode/evaluasi/pelaporan; penerimaan jurnal belum dapat dijamin.

## Urutan kerja singkat

**Diagnostik/stabilitas development → perbaikan recipe terkontrol → mutu GAN baru → baseline/tuning/GAN dengan grid disepakati → evaluasi diperkuat/final → naskah.** Penulisan metode dan hasil V3 yang sudah tersedia dapat dilakukan paralel dengan eksperimen. Revisi detector dan kontrol durasi crop menjadi penelitian tambahan setelah eksperimen inti cukup stabil.

Notebook training baru sudah tersedia. **Mulai 20 → 21**, kemudian pilot 22; periksa diagnosis/pilot di 27. Grid lengkap tetap diperlukan sebelum final 26. GAN baru dan penguatan evaluasi belum diselesaikan oleh implementasi V4 real-only.
