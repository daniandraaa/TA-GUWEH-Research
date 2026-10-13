# Langkah setelah notebook 36 V6 terhenti

**Status 5 Oktober 2026: notebook 37–40 sudah selesai dan diaudit, lima fold PASS. Tidak perlu menjalankan ulang training.** [Analisis hasil](ANALISIS_HASIL_V7_20261005.md) · [fase classifier berikut](FASE_CLASSIFIER_SETELAH_V7_20261005.md). `status.next_step` di CLI adalah teks urutan setup yang tetap; lihat status setiap fold serta audit untuk kemajuan terbaru. Petunjuk di bawah menjadi referensi alur yang telah selesai.

**Urutan saat implementasi: 37 → 38 → 39, kemudian 40 setelah PASS.** V6 fold 1 selesai, tetapi gagal screening mutu; notebook 36 tidak diteruskan. V7 adalah revisi objective GAN berdasarkan kegagalan tersebut. Hasil, kode dan notebook V6 tetap dipertahankan. Training data asli tetap Anda jalankan manual.

| Urutan | Notebook | Tindakan |
|---|---|---|
| 1 | [37 — Persiapan target dan smoke](../../notebooks/37_Persiapan_Target_Distribusi_dan_Smoke_GAN_V7.ipynb) | Setup → prepare → smoke MPS → status |
| 2 | [38 — Pilot fold 1](../../notebooks/38_Pilot_Fold1_GAN_Loss_Distribusi_V7.ipynb) | Training 100 epoch → generate → QC → status |
| 3 | [39 — Visualisasi/perbandingan](../../notebooks/39_Perbandingan_Visualisasi_Mutu_GAN_V6_V7.ipynb) | Report → tampilkan grafik/tabel V6 versus V7 |
| 4 | [40 — Fold 2–5](../../notebooks/40_GAN_Fold2_sampai5_Setelah_Pilot_V7_Lulus.ipynb) | Hanya setelah pilot V7 PASS dan hasil notebook 39 diperiksa |

**Kerjakan 37–39 dahulu, lalu minta audit hasil.** Tidak perlu membuka 36 lagi untuk V7. Tidak perlu mengulang 33–35. Jika V7 FAIL, notebook 39 tetap dapat menghasilkan laporan perbandingan; jangan menjalankan 40 sebelum pilot lolos.

**Persiapan lima fold telah selesai tanpa training.** Sel `prepare` pada 37 memverifikasi hasil yang tersedia dan melewatinya; smoke MPS tetap perlu Anda jalankan.

Run: `20261004_fit_distribution_gan_v7`. Kode: `research_v7/`; runner: `_v7_runner.py`; output: `results/research_v7/20261004_fit_distribution_gan_v7/`.

## Cara menjalankan

1. Pilih kernel `.venv/bin/python` project di VS Code/Jupyter.
2. Buka 37 dan jalankan sel satu per satu. `prepare` menyalin daftar pasien/window/skala/fitur QC V6 yang sudah terverifikasi, lalu menghitung target distribusi **hanya dari pasien GAN-fit**. Tidak melatih generator. Target tidak memakai pasien referensi QC atau outer validation/test.
3. Jalankan `smoke`. Generator/discriminator memakai MPS dengan batch 16; statistik dan FFT loss memakai CPU secara eksplisit. Smoke menguji satu update G/D disposable dan backward fidelity menuju generator. Tidak ada bobot smoke yang dipakai sebagai checkpoint awal pilot.
4. Jika smoke selesai, buka 38. Sel `train` memulai V7 dari bobot acak dengan seed yang sama seperti V6. **Tidak resume dari checkpoint V6.** Log epoch memperlihatkan adversarial, fidelity total dan masing-masing loss amplitudo/roughness/korelasi/spektrum.
5. Setelah 100 epoch, jalankan `generate` dan `qc`, kemudian 39. QC memakai kode dan ambang V6 yang sama; hasil bisa PASS atau FAIL. Semua 192 sampel dianalisis, tidak difilter/dipilih sesudah generasi.
6. Hasil notebook 39 menjadi dasar keputusan apakah V7 memenuhi syarat untuk 40. PASS belum membuktikan manfaat classifier. Fase classifier K4 real-only/kontrol augmentasi/GAN tetap sesudah seluruh generator bermutu.

Jika Interrupt, tunggu proses berhenti lalu ulang sel pada device yang sama. Resume dari epoch lengkap terakhir menyimpan model, optimizer, RNG global, dataset, loader dan latent. Epoch yang terputus diulang. Hasil lengkap digunakan ulang dengan verifikasi hash. Jangan menjalankan dua training sekaligus.

Jika smoke gagal, salin ekor log yang ditampilkan. Jangan mengubah batch/device atau checkpoint di tengah run. Perubahan recipe/runtime/device memerlukan run-id baru; jangan mengedit `protocol.json` agar cocok. Dukungan backward MPS dengan loss CPU perlu dikonfirmasi oleh smoke manual ini; pengujian otomatis yang dilakukan memakai CPU sintetis.

## Apa yang diubah

Arsitektur G/D sama persis dengan V6. Pasien fit/reference, skala amplitudo, sampling kelas→pasien→window, RNG/seed, LR 1e-4, batch 16, 100 epoch, Adam/betas, gradient clip 5, generasi 64 window/kelas dan gate QC juga sama. Dengan demikian, revisi ini menguji **paket perubahan objective**, bukan sekaligus mengganti model dan menambah anggaran epoch.

Moment matching mean/SD pada minibatch V6 diganti dengan target quantile kelas tetap pada 0,10/0,25/0,50/0,75/0,90. Target dihitung dari maksimal tiga window per pasien GAN-fit; total bobot setiap pasien satu. Quantile sintetis setiap kelas dibandingkan dengan target, dengan skala IQR fit dan SmoothL1. Empat keluarga diberi bobot 1 masing-masing:

- **Amplitudo:** log RMS dan log rentang p95−p05 pada setiap lead.
- **Roughness:** log rasio derivative RMS terhadap signal RMS pada lag 1/4/16/64.
- **Korelasi:** 66 pasangan lead pada 12 kanal, sehingga hubungan precordial ikut diperiksa.
- **Spektrum:** log daya relatif tujuh band 0,5–1000 Hz dari Welch window 500/1000/2000 sampel, overlap 50%. Resolusi frekuensinya 4/2/1 Hz. Floor log power 1e-6 ditetapkan sebelum training.

Loss generator = adversarial + empat loss keluarga. IQR floor: amplitudo 0,25; roughness 0,25; korelasi 0,10; spektrum 0,50. Angka ini pilihan engineering awal yang dicatat, belum hyperparameter optimal atau standar klinis. Tidak memperbarui target dengan fake/reference selama training, tidak memasangkan random fake dengan waveform real, dan tidak menganggap generator ini reproduksi Parallel WaveGAN.

Statistik/FFT dilakukan pada CPU tanpa `detach` untuk jalur sintetis, lalu scalar loss kembali ke device generator. Ini menjaga operasi spektrum dari ketergantungan FFT kompleks MPS; konsekuensinya ada biaya transfer/CPU yang perlu diukur. Implementasi dengan PyTorch terpasang diuji pada sinyal sintetis. Runtime dipatok dalam manifest run.

QC tetap membaca 37 pasien referensi mutu per fold, menggunakan fitur native dan threshold yang sama seperti V6. Keluarga loss dan QC tumpang tindih karena masalah yang ditargetkan sama; lolos QC bukan validasi independen seluruh informasi ECG. Perlu evaluasi classifier dan generalisasi berikutnya. Tidak tersedia validasi ahli ECG atas PVC/kondisi anatomi sintetis.

## Output yang diperiksa pada notebook 39

- `history.csv`: loss keluarga, exposure pasien/kelas, gradient norm dan biaya per epoch.
- `quality.json`: finite/copy/diversity dan keluarga fitur yang gagal.
- `comparison_v6_v7.csv` dan `figures/comparison_v6_v7.png`: fraksi fitur gagal pada fold yang sama. Angka ini **bukan akurasi classifier**.
- Waveform 12 lead, distribusi frekuensi dan nearest-neighbour native ke GAN-fit.
- `REPORT.md`: PASS/FAIL serta batas interpretasi.

V6 hanya mempunyai generator lengkap fold 1, sehingga pembanding V6 versus V7 tersedia pada fold 1. Fold 2–5 V7 dilaporkan dengan QC masing-masing setelah pilot PASS; tidak membuat-buat hasil pembanding V6 yang belum dijalankan.

Disk guard tetap 2 GiB bebas ditambah cadangan 256 MiB; tidak membuat salinan cache besar. Checkpoint dan 192 window float32 menambah sekitar 0,1 GiB per fold. Jangan menghapus hasil V6 untuk menggantinya dengan V7.

Alternatif terminal dari root project:

```bash
.venv/bin/python -m research_v7 prepare
.venv/bin/python -m research_v7 smoke --device mps
.venv/bin/python -m research_v7 train --fold 1 --device mps
.venv/bin/python -m research_v7 generate --fold 1 --device mps
.venv/bin/python -m research_v7 qc --fold 1
.venv/bin/python -m research_v7 report --fold 1
.venv/bin/python -m research_v7 status
```

[Dasar riset dan desain V7](REVISI_OBJECTIVE_GAN_V7_20261004.md). Skor 0,9 atau penerimaan Q1 tidak menjadi kriteria untuk mengubah label, memilih seed test, atau melemahkan screening.
