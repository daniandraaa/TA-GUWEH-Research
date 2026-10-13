# Riset dan fase peningkatan fitur — 6 Oktober 2026

**Hasil aktual — 7 Oktober 2026:** pengukuran LUDB PASS, ekstraksi 230 pasien lengkap, dan tiga arm fitur sudah menyelesaikan seluruh lima fold (150 fit). Mean fold F1 **QRS/RR 0,4210; morfologi 0,5555; morfologi + RR 0,5532**. Tambahan RR belum memberi kenaikan konsisten; pemisahan cusp–other dan identitas/agregasi beat tetap prioritas. Empat arm encoder/fusion belum dijalankan karena cache V10 tidak tersedia. [Analisis, ketidakpastian, grafik dan tindakan berikutnya](ANALISIS_HASIL_FITUR_V11_20261007.md). Rencana dan status sebelum run di bawah dipertahankan sebagai riwayat.

Usulan lebar QRS, RR dan identitas lead layak diuji, tetapi manfaatnya belum diketahui pada tiga target project. Prioritasnya adalah memastikan informasi lokalisasi benar-benar masuk ke model, mengurangi overfitting, lalu membuktikan kenaikan pada pasien yang berbeda. Fase kode baru V11 tersedia dengan [panduan manual 57–61](PANDUAN_MANUAL_V11.md); training ECG asli tetap dijalankan pengguna.

## Bukti project yang mendasari keputusan

V10 selesai: mean fold/seed macro-F1 random frozen 0,4415 versus pretrained frozen 0,3744; tail fold1/seed42 0,3928. Notebook 56 lengkap 5/5 sel, tanpa error. Enam fit yang direplay menghasilkan prediksi sama; 24 pemeriksaan encoder pada tiga pasien training lulus. Belum ditemukan bukti kerusakan cache/CPU–MPS pada sampel tersebut.

Outer fold1/seed42: training/evaluation macro-F1 random **0,8157/0,4247**, pretrained **0,7635/0,4454**, tail **0,7703/0,3928**. Gap cukup besar pada fit yang diperiksa, sehingga regularisasi menjadi prioritas. Ini tidak membuktikan semua fit identik atau bahwa fitur baru pasti menyelesaikan masalah.

Encoder V8/V9 sekitar 0,53 bukan batas maksimum yang telah terbukti. Skor lebih tinggi di V3/V4 berasal dari protokol/window/training yang berbeda; jangan membandingkan tanpa kontrol pasien/fold/preprocessing. Eksperimen baru memakai pembagian V10 yang tetap agar manfaat fitur versus kontrol head dapat dinilai.

## Perbaikan selain QRS, RR dan lead

| Prioritas | Apa yang diperbaiki | Mengapa penting | Status |
|---|---|---|---|
| 1 | Identitas beat PVC versus sinus dan pemilihan beat representatif | Morfologi sinus yang dirata-ratakan dapat menutupi morfologi aritmia | Belum terselesaikan; semua kandidat diberi status PVC belum terverifikasi |
| 2 | Deteksi onset/offset, ketepatan sampling, denominator kegagalan dan pratinjau | Fitur durasi salah dapat merusak classifier meskipun namanya klinis | Implementasi V11 + benchmark LUDB tersedia, hasil nyata menunggu manual |
| 3 | Polaritas/amplitudo antarlead, rasio R/S dan deskriptor transisi | Lokalisasi bergantung pada pola antarlead, bukan nomor urut lead saja | 126 fitur morfologi dan 132 dengan RR tersedia |
| 4 | Regularisasi dan kontrol dimensionalitas | Sampel pasien kecil dibanding representasi 768 dimensi; gap train–validation besar | Head L2 dengan 9 C inner tersedia; PCA/seleksi fitur belum ditambahkan |
| 5 | Agregasi denyut per pasien yang tahan variasi | Median/IQR adalah kontrol sederhana; beat informatif dapat menjadi minoritas | Median/IQR tersedia; attention/MIL morphology-aware menjadi opsi sesudah hasil V11 |
| 6 | Pemisahan dua kelompok RVOT dan analisis kasus gagal | Macro-F1 naik bisa tetap menurunkan sensitivitas satu kelompok | Recall tiap kelas dan perbandingan paired tersedia; inspeksi error sesudah run |
| 7 | Evaluasi independen, ketidakpastian dan sumber pretraining | Metode berkembang dari development; historical test sudah dilihat, overlap pretraining belum pasti | OOF/CI conditional tersedia; external origin validation/nested confirmatory belum selesai |

**Yang paling mendasar adalah prioritas 1.** Pengenalan kandidat QRS berhasil tidak sama dengan pengenalan PVC. Tanpa anotasi beat lokal, otomatis memilih beat terlebar atau berdasarkan confidence classifier dapat menimbulkan kesalahan baru. V11 tidak membuat pseudo-label PVC dari aturan durasi. Setelah kualitas dan manfaat fitur diketahui, opsi berikut adalah detector PVC yang tervalidasi pada sumber beranotasi beat, disertai audit domain lokal; tetap tidak mengklaim verifikasi klinis beat lokal.

Prioritas 3 mempertahankan informasi amplitudo relatif dan polaritas. Fitur V2S/V3R diperlakukan sebagai kandidat input, bukan rule yang langsung menentukan semua kelas. Nomor lead I–V6 adalah identitas sensor; menambahkan angka konstan pada semua pasien tidak menambah variasi yang bisa membedakan label. Lead-aware embedding pada jaringan sudah pernah dicoba V4; tambahan yang diuji sekarang adalah informasi morfologi per lead dan fusion dengan cache encoder.

Prioritas 4 menguji apakah representasi masih dapat digunakan dengan head berregularisasi lebih kuat. PCA, supervised feature selection atau jaringan fusion kecil dapat menjadi perluasan, tetapi harus di-fit/dipilih hanya pada training/inner split. Jangan menentukan jumlah komponen atau fitur dari seluruh development lalu melaporkan outer fold sebagai evaluasi bebas kebocoran.

Kalibrasi probabilitas, abstention dan penjelasan fitur dapat menyusul model yang stabil. SHAP/attention memberi penjelasan komputasional; tanpa pemeriksaan klinis, jangan menyebut hasilnya alasan diagnosis dokter. Tidak ada tambahan klaim kausal dari ranking fitur.

## Apa yang dapat diambil dari referensi

**Paper asal:** [Zheng dkk., 2022, A High Precision Machine Learning-Enabled System for Predicting Idiopathic Ventricular Arrhythmia Origins](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2022.809027/full). Paper memakai kompleks PVC/VT yang dipilih operator dan disejajarkan terhadap puncak R lead II, serta menguji window pendek di sekitar QRS dengan ExtraTrees pada informasi voltage. Implikasi untuk project: seleksi beat dan alignment layak diprioritaskan sebelum menambah arsitektur. Ini inferensi metodologis, bukan reproduksi hasil atau perbandingan pasien yang setara; cohort/label/protokol paper dan project perlu dibedakan.

**Rasio antarlead:** [Yoshida dkk., 2014, V2S/V3R index](https://pubmed.ncbi.nlm.nih.gov/24612087/). Studi ini menguji pembedaan asal LVOT versus RVOT; dasar untuk menguji rasio morfologi antarlead tersedia. Bukti tersebut tidak otomatis berlaku untuk pembedaan RVOT cusp versus RVOT other atau menghasilkan akurasi sama pada project.

**Pengukuran:** [LUDB 1.0.1 / PhysioNet](https://physionet.org/content/ludb/1.0.1/) menyediakan batas QRS manual oleh kardiolog, 200 rekaman 12 lead/500 Hz. Ini relevan untuk memeriksa algoritme onset/offset ketika ahli ECG lokal belum tersedia. Program menguji semua rekaman tanpa tuning threshold pada LUDB. Benchmark tersebut bukan external validation classifier tiga kelompok.

**Optimisasi head:** [dokumentasi resmi scikit-learn, logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression). C mengendalikan inverse regularization strength. Implementasi mengikuti API environment lokal 1.9.0, menggunakan LBFGS, L2 dan class weights yang dihitung dari training. Nilai LR/batch tidak dituning untuk solver ini.

## Fase konkret dan keputusan setelah run

1. **57:** persiapan/smoke sintetis, tanpa fitting ECG asli. Jika source/cache/fold tidak cocok, diagnosis terlebih dahulu.
2. **58:** benchmark pengukuran manual. PASS adalah gate rekayasa yang prespecified; FAIL tetap hasil penelitian dan perlu diagnosis. Jangan menurunkan standar setelah melihat skor.
3. **59:** ekstraksi, audit missingness dan pratinjau. Tidak menghapus pasien bermasalah agar skor naik. Konversi native ke ms, RR kontinu dan skala bersama ditelusuri di setiap record.
4. **60:** pilot fold1 seluruh tujuh arm. Periksa convergence/identitas; hasil fold1 tidak dipakai sendirian untuk memilih pemenang.
5. **61:** lengkapi lima fold dan audit. Bandingkan fitur vs head saja dan fusion vs encoder-head yang sama. Perhatikan macro-F1, recall cusp/other, gap, lima delta fold, pooled OOF dan CI conditional.
6. **Sesudah hasil V11:** jika fusion membantu, pertimbangkan integrasi fitur ke CNN–Transformer dengan kontrol yang setara. Jika tidak, diagnosis apakah pengukuran, dominasi sinus, skala, dimensionalitas atau pemisahan target menjadi masalah; jangan otomatis memperluas grid/GAN. Siapkan evaluasi pasien independen untuk klaim generalisasi sebelum final publication claims.

Belum ada hasil V11 dan belum ada model final baru. Q1 tidak mempunyai ambang tunggal macro-F1 ≥0,9. Klaim paper harus dibangun dari pertanyaan/kontribusi yang jelas, kontrol yang adil, validasi independen, ketidakpastian dan batas klinis yang disampaikan secara jujur. Akurasi tinggi pada problem/data/protokol yang berbeda tidak menjadi bukti bahwa 0,9 tersedia pada tiga kelompok project ini.
