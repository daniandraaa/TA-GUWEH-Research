# Menjalankan eksperimen revisi TA

**V12 sudah selesai dijalankan — 7 Oktober 2026:** **62–67 lengkap**, 40 fit CV dan satu deployment terverifikasi. Mean fold F1 morfologi+RR **0,7897**, tetapi eksternal MIT-BIH **0,3998/gate FAIL**. **68 melewati ekspor sesuai gate**; tidak ada masukan lokal/model origin baru. **Tidak perlu mengulang 62–68 atau memaksa gate PASS.** Langkah berikut ialah perbaikan detector/pusat beat, lalu generalisasi classifier/calibration pada run terpisah; belum ada notebook eksperimen berikutnya pada audit ini. [Diagnosis dan prioritas](docs/revisi/ANALISIS_HASIL_V12_20261007.md). [Panduan V12](docs/revisi/PANDUAN_MANUAL_V12.md) dipertahankan sebagai referensi pelaksanaan yang telah selesai.

**Kajian literatur terbaru — 7 Oktober 2026:** baca [kajian CNN–Transformer/PVC dan fase penelitian berikut](docs/revisi/KAJIAN_LITERATUR_CNN_TRANSFORMER_PVC_20261007.md). Prioritas ilmiah ialah identitas beat, masukan terukur dan hubungan antarlead, kemudian pembandingan pasien terkontrol. Fase pertama kini tersedia sebagai V12; belum ada training baru atau pemantauan rutin. Notebook yang sudah selesai tidak perlu diulang untuk membaca kajian.

**Status terbaru — 7 Oktober 2026:** fase V10/diagnosis 56 dan pengukuran/ekstraksi V11 telah dijalankan. **Notebook 60/61 selesai untuk tiga arm fitur**, 150 fit terverifikasi; mean fold macro-F1 QRS/RR **0,4210**, morfologi **0,5555**, morfologi + RR **0,5532**. **Tidak mengulang notebook 00–59 atau tiga arm fitur pada 60/61.** Empat arm encoder/fusion masih menunggu pemulihan cache V10 yang sesuai audit. Baca [hasil dan langkah berikutnya](docs/revisi/ANALISIS_HASIL_FITUR_V11_20261007.md) serta [panduan V11](docs/revisi/PANDUAN_MANUAL_V11.md); arahan fase lama di bawah adalah riwayat.

**Notebook 46–50 sudah selesai dan diaudit — 6 Oktober 2026.** Seluruh **30/30 kasus V9 = 20 fit baru + 10 referensi** valid. Encoder F1 naik sedikit tetapi recall other turun; Hierarchical turun. [Hasil dan tindak lanjut](docs/revisi/ANALISIS_HASIL_V9_20261006.md). **Training V9 tidak perlu diulang.** Tahap berikut adalah audit kecocokan pretrained ECG/adapter dan protokol pembanding fine-tuning; belum ada notebook atau training pretrained baru.

**Mulai fase V9 dari notebook 46 → 47 → 48 → 49 → 50.** Persiapan/diagnosis selesai, **10/30 referensi lengkap dan 0/20 fit baru**. Jalankan smoke pada 46, pilot pada 47, lalu masing-masing model 48/49 dan audit 50. [Panduan manual V9](docs/revisi/PANDUAN_MANUAL_V9.md) · [rencana representasi/fine-tuning](docs/revisi/RISET_REPRESENTASI_DAN_FINE_TUNING_20261006.md). Tidak mengulang V8; tidak ada training/final/test baru dijalankan otomatis.

**Notebook 41–45 sudah selesai — 6 Oktober 2026:** seluruh **120 fit V8** dan hasil analisis lolos audit. [Analisis lengkap](docs/revisi/ANALISIS_HASIL_V8_20261006.md). GAN belum menjadi perbaikan stabil terhadap kedua kontrol dan kedua recall RVOT. **Training ini tidak perlu diulang.** Berikutnya riset/protokol stabilitas epoch dan representasi real-only pada run baru; [urutan fase](docs/revisi/FASE_SETELAH_V8_20261006.md). Notebook 46–50 kini tersedia sebagai fase V9, lihat pembaruan terbaru di atas.

**Mulai V8 dari notebook 41 → 42**, lalu 43 → 44 → 45 sesudah pilot proses selesai. Pilih kernel `.venv`, jalankan sel satu per satu; satu proses GPU pada satu waktu. V8 membandingkan real-only, real augmentation dan GAN augmentation dengan epoch inner dan update budget setara. [Panduan lengkap](docs/revisi/PANDUAN_MANUAL_V8.md). Training data asli belum dijalankan oleh Codex.

**V7/notebook 37–40 sudah selesai — 5 Oktober 2026:** seluruh GAN lima fold serta QC native lolos audit; training ini tidak perlu diulang. [Analisis hasil dan bukti](docs/revisi/ANALISIS_HASIL_V7_20261005.md) · [fase classifier berikut](docs/revisi/FASE_CLASSIFIER_SETELAH_V7_20261005.md). Selanjutnya perbandingan K4 real-only/kontrol augmentasi real/real+GAN pada validation pasien real. Endpoint baru tersedia pada V8/notebook 41–45; tidak menggunakan 37–40 untuk training classifier. [Panduan V7](docs/revisi/PANDUAN_MANUAL_V7.md) dipertahankan sebagai referensi eksekusi.

> **Fase V4 telah selesai — 3 Oktober 2026:** notebook 20–27 tanpa error tersimpan; diagnosis 45, CV 450, hasil stability 75 (25 memakai ulang CV), final 25 dan test 25. Mulai tindak lanjut dari [analisis hasil V4](docs/revisi/ANALISIS_HASIL_V4_20261003.md) dan daftar kesalahan development, lalu ablation pada run baru. [Panduan V4](docs/revisi/PANDUAN_MANUAL_V4.md) tetap menjadi referensi eksekusi; notebook 20–26 tidak perlu diulang.

**Riwayat V6/notebook 33–35 selesai:** pilot fold 1 gagal screening mutu sehingga 36 berhenti sebelum fold 2–5. [Diagnosis](docs/revisi/ANALISIS_PILOT_GAN_V6_20261004.md) · [panduan V6 sebagai referensi](docs/revisi/PANDUAN_MANUAL_V6.md). Pertahankan hasil/code V6; tahap berikut memakai run V7 baru.

**V5/notebook 29–32 selesai — 4 Oktober 2026:** kedua smoke MPS, 30 fit baru dan 30 kontrol terverifikasi. K8 belum memperbaiki hasil; **pertahankan K4**. [Analisis lengkap V5](docs/revisi/ANALISIS_HASIL_V5_20261004.md). Tidak perlu training ulang 29–31. [Panduan V5](docs/revisi/PANDUAN_MANUAL_V5.md) menjadi referensi eksekusi yang selesai. Notebook 28 lengkap pada 230 development dan 12 pratinjau.

> **Analisis hasil V2 tersedia:** 270 CV dan seluruh final/test sudah lengkap. Buka [Notebook 10](notebooks/10_Analisis_Hasil_V2.ipynb) atau [laporan 1 Oktober 2026](docs/revisi/ANALISIS_HASIL_V2_20261001.md) untuk tabel, diagnosis per kelas, variasi seed, dan prioritas lanjutan.

**Alur V4 yang sudah selesai:** 20 persiapan → 21 diagnosis → 22 pilot → 23–25 full grid → 26 stability/final/test → 27 visualisasi. Dua CNN–Transformer baru dan bag pasien diuji bersama tiga pembanding. [Kajian metode/protokol/anggaran](docs/revisi/RISET_METODE_DAN_PROTOKOL_V4_20261002.md). Target 0,9 belum terukur; GAN terkoreksi, nested evaluation dan validasi eksternal masih terbuka. Penulisan Bab III–V dapat memakai hasil V3 yang sudah tersimpan.


> **Riwayat V2 yang sudah selesai:** [Notebook 06 — Audit sumber dan preprocessing](notebooks/06_Audit_Sumber_dan_Preprocessing_V2.ipynb), kemudian 07–09. Sumber menegaskan 2.000 Hz; tiga kelas dikoreksi namanya sesuai anatomi dengan anggota tetap. [Panduan V2 dan batas interpretasi](docs/revisi/PANDUAN_REVISI_V2.md).


## Mulai secara manual

**Setelah menjalankan kelima notebook eksperimen, buka [05_Visualisasi_Proses_dan_Hasil.ipynb](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>).** Gunakan VS Code atau Jupyter, pilih kernel Python `.venv` project, lalu jalankan **sel satu per satu dengan Shift+Enter**. Notebook ini membaca hasil yang sudah ada; tidak menjalankan training atau GPU. Sembilan belas grafik PNG hasil pemeriksaan tersedia di `results/finetune_revision/20260929_grid9_v1/visualisasi/`.

Jika perlu menelusuri alur training: di [notebook 00](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/00_Mulai_di_Sini.ipynb>), tahapnya adalah pengaturan → status → data development → fold/scaler → GAN → pemeriksaan selesai. EDA dan DWT sudah tersedia; runner memakai hasil preprocessing tersebut. Sesudah GAN kelima fold lengkap, urutannya sebagai berikut.

| Urutan | File | Yang dijalankan |
|---|---|---|
| 00 | [Mulai di sini](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/00_Mulai_di_Sini.ipynb>) | Persiapan dan GAN bersama untuk lima fold |
| 01 | [Tuning Encoder](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/01_Tuning_Encoder.ipynb>) | 45 training CV CNN-Transformer Encoder |
| 02 | [Tuning PatchTST](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/02_Tuning_PatchTST.ipynb>) | 45 training CV CNN-PatchTST |
| 03 | [Tuning Hierarchical](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/03_Tuning_Hierarchical.ipynb>) | 45 training CV CNN-Hierarchical Transformer |
| 04 | [Final dan evaluasi](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/04_Final_dan_Evaluasi.ipynb>) | Seleksi, baseline CV, retraining final, inferensi, SHAP, laporan |
| 05 | [Visualisasi proses dan hasil](</Users/daniandraaa/Desktop/TA GUWEH/notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb>) | Grafik data, GAN, grid CV, training, test, SHAP; tanpa training |

Notebook 01–03 khusus tuning per arsitektur. Sesudah ketiganya selesai (135/135 training CV), notebook 04 mengatur seluruh tahap final dari satu tempat. Seleksi menggunakan hasil CV development. Sebelum inferensi, notebook memeriksa bahwa seluruh enam training final (tiga fine-tuning dan tiga baseline) selesai.

Gunakan satu proses training GPU pada satu waktu. Output epoch muncul langsung di bawah sel dan disalin ke `results/finetune_revision/20260929_grid9_v1/manual_logs/`. Tombol **Interrupt/Stop** menghentikan proses training anak; menjalankan kembali sel yang sama dengan `--resume` melanjutkan akhir epoch terakhir yang tersimpan. Epoch yang terputus di tengah akan dijalankan ulang.

Seluruh notebook memakai `RUN_ID = "20260929_grid9_v1"`, `DEVICE = "mps"`, dan `.venv/bin/python` project. Checkpoint yang belum selesai harus dilanjutkan pada jenis perangkat yang sama. Pertahankan kode, konfigurasi, dan input selama eksperimen ini.

### Status terkini

Pada audit 30 September 2026, GAN lima fold dan final, 135/135 CV fine-tuning, 15/15 CV baseline, enam final, enam test, dan dua analisis SHAP sudah tersedia. Hasil dan langkah penulisan dijelaskan pada [analisis eksperimen](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/ANALISIS_HASIL_EKSPERIMEN.md>).

`pipeline.state=stopped` masih mencatat penghentian proses otomatis pada 29 September; jumlah artefak dalam `status.json` mencakup training manual yang sudah selesai. Tidak perlu menjalankan ulang tahap training untuk memakai notebook visualisasi.

### Membuka dengan Jupyter dari Terminal

Jika Anda memilih Jupyter, jalankan:

```bash
cd "/Users/daniandraaa/Desktop/TA GUWEH"
source .venv/bin/activate
python -m jupyterlab notebooks/05_Visualisasi_Proses_dan_Hasil.ipynb
```

Perintah ini membuka notebook visualisasi. Jika memilih VS Code, buka folder project lalu file notebook 05 dan pilih interpreter `.venv/bin/python` sebagai kernel.

[Peta file](</Users/daniandraaa/Desktop/TA GUWEH/docs/PETA_FILE.md>) menjelaskan letak data, kode aktif, dokumen, arsip, dan hasil. Notebook eksperimen lama berada di `arsip/notebooks_lama/`; alur revisi memakai notebook 00–05.

## Protokol dan checkpoint

Grid: learning rate `[1e-4, 5e-4, 1e-3]` × batch size `[16, 32, 64]`. Ketiga arsitektur berbagi definisi model baseline, manifest pasien, dan evaluasi patient-level. GAN dan scaler dibuat khusus untuk training fold. Hold-out dibaca setelah seleksi konfigurasi dan training final.

Lingkungan `.venv` project sudah menyediakan dependensi termasuk kernel notebook. Jalankan perintah dari folder project. Konfigurasi default menggunakan satu seed final, yaitu 42. Semua artefak disimpan di `results/finetune_revision/20260929_grid9_v1/`.

Pantau hasil:

```bash
.venv/bin/python scripts/run_revision.py status
```

`pipeline.state` mencatat runner otomatis yang telah dihentikan. Jumlah GAN/CV/final/test pada `status.json` dihitung dari artefak yang tersimpan, termasuk hasil eksekusi manual. Log sebelum penghentian berada pada `pipeline.log`; log dari notebook berikutnya berada pada `manual_logs/`.

Setiap training memiliki `config.json`, `history.csv`, best checkpoint, prediksi validation, serta `metrics.json` sebagai marker selesai. `last.pt` menyimpan optimizer/RNG per epoch selama training; setelah selesai file itu dipadatkan. Setelah interupsi, gunakan perintah yang sama dengan `--resume`. Kasus lengkap dilewati. Resume melanjutkan dari akhir epoch terakhir yang tersimpan, bukan dari pertengahan minibatch.

Jika kode, konfigurasi, atau input berubah, gunakan `--run-id` baru. Runner menolak pencampuran identitas eksperimen. Semua sinyal development disimpan sebagai memory map bersama; normalisasi dilakukan per sampel menggunakan scaler fold untuk mengurangi duplikasi disk.

## Menjalankan per tahap atau model

Perintah berikut adalah rujukan untuk mengulang atau melanjutkan tahap tertentu jika dibutuhkan. Semua hasil utama saat ini sudah lengkap; menjalankannya bukan syarat untuk melihat notebook 05. Jika memilih Terminal, jalankan baris satu per satu.

```bash
cd "/Users/daniandraaa/Desktop/TA GUWEH"
.venv/bin/python scripts/00_prepare_data.py
.venv/bin/python scripts/01_prepare_folds.py
.venv/bin/python scripts/02_prepare_fold_augmentation.py --device mps --all-folds --resume
.venv/bin/python scripts/03_tune.py --model encoder --device mps --all-scenarios --all-folds --resume
.venv/bin/python scripts/03_tune.py --model patchtst --device mps --all-scenarios --all-folds --resume
.venv/bin/python scripts/03_tune.py --model hierarchical --device mps --all-scenarios --all-folds --resume
.venv/bin/python scripts/04_select_best.py --all-models
.venv/bin/python scripts/run_revision.py baseline --device mps --resume
.venv/bin/python scripts/04_select_best.py --all-models --baseline
.venv/bin/python scripts/05_train_final.py --model encoder --device mps --seed 42 --resume
.venv/bin/python scripts/05_train_final.py --model patchtst --device mps --seed 42 --resume
.venv/bin/python scripts/05_train_final.py --model hierarchical --device mps --seed 42 --resume
.venv/bin/python scripts/05_train_final.py --all-models --baseline --device mps --seed 42 --resume
.venv/bin/python scripts/06_evaluate_test.py --model encoder --device mps --seed 42
.venv/bin/python scripts/06_evaluate_test.py --model patchtst --device mps --seed 42
.venv/bin/python scripts/06_evaluate_test.py --model hierarchical --device mps --seed 42
.venv/bin/python scripts/06_evaluate_test.py --all-models --baseline --device mps --seed 42
.venv/bin/python scripts/07_explain.py --device mps
.venv/bin/python scripts/07_explain.py --device mps --baseline
.venv/bin/python scripts/08_compare.py
```

Untuk mengerjakan satu kasus saja pada run baru, gunakan `--model encoder --scenario S01 --fold 1 --resume` pada `scripts/03_tune.py`. Tahap lengkap di run ini akan dilewati oleh runner.

Baseline CV juga bisa dijalankan per model menggunakan `scripts/run_revision.py baseline --model encoder --device mps --resume`, lalu ganti model menjadi `patchtst` dan `hierarchical`. Seleksi final memerlukan grid ketiga model selesai agar model utama ditetapkan sebelum test.

## Keluaran laporan

- `protocol.json`: konfigurasi tetap, sembilan skenario, fingerprint kode/input, serta versi lingkungan.
- `folds/manifest.json`: ID training dan validation lima fold.
- `folds/fold_*/scaler_provenance.json`: asal pasien dan statistik scaler.
- `augmentation/fold_*/manifest.json`: asal pasien GAN, seed, dan distribusi kelas.
- `<model>/cv_fold_results.csv` dan `cv_summary.csv`: 45 hasil fold dan sembilan ringkasan.
- `<model>/best_config.json` dan `selected_model.json`: pilihan dari CV.
- `<model>/final/seed_42/`: checkpoint final, konfigurasi, dan history.
- `<model>/test/seed_42/`: metrik, prediksi pasien, serta analisis sublokasi.
- `<model>/explain/`: SHAP, importance sadapan, dan contoh kontribusi waktu.
- `comparison/HASIL_REVISI.md`: angka dan grafik untuk pembahasan hasil.
- `progress.json` / `status.json`: tahap aktif serta jumlah training yang sudah selesai.
- `manual_logs/`: perintah dan output setiap tahap yang Anda jalankan dari notebook.
- `visualisasi/`: 19 grafik PNG dari notebook 05, termasuk diagnostik GAN pada semua fold dan hubungan sadapan ekstremitas.

[Draf metodologi Bab III](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/REVISI_METODOLOGI_BAB_III.md>) dan [fase revisi](</Users/daniandraaa/Desktop/TA GUWEH/docs/revisi/FASE_REVISI_FINE_TUNING.md>) berada di `docs/revisi/`. [Proposal PDF](</Users/daniandraaa/Desktop/TA GUWEH/docs/proposal/PROPOSAL TA.pdf>) berada di `docs/proposal/`; isi PDF belum diedit karena source editable belum tersedia. Hold-out historis dan asumsi sampling rate dicantumkan secara eksplisit di protokol/laporan.

Verifikasi:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_revision.py --device mps
```
# V3 setelah audit proposal

Alur V3 dimulai dari persiapan **11**, full grid **12–14**, comparator/seleksi/final/test **15**, analisis **18** dan SHAP primer V3 **19**. Seluruh tahap itu sekarang selesai melalui run manual pengguna. Notebook **16** menjelaskan model V2; **17** juga sudah menyelesaikan benchmark 75 rekaman. Notebook 11–19 memisahkan tahap/model dan dijalankan manual.

Panduan: [PANDUAN_MANUAL_V3](docs/revisi/PANDUAN_MANUAL_V3.md). Audit 20 referensi: [AUDIT_DAFTAR_PUSTAKA_20261001](docs/revisi/AUDIT_DAFTAR_PUSTAKA_20261001.md). Kode baru di `research_v3/`; kode/hasil V1–V2 dipertahankan.

Kandidat otomatis tidak dianggap PVC terverifikasi. Tanpa ahli ECG baru, benchmark anotasi beat publik INCART sudah menguji detector dan mengungkap kegagalan pada beberapa rekaman; ia tidak memvalidasi asal anatomi tiga kelas. Penguatan stabilitas classifier, kontribusi ilmiah dan validasi tetap diperlukan untuk target publikasi.
