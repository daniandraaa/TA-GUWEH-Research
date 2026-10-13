# Menjalankan GAN V6 setelah V5 selesai

**Hasil manual terbaru, 4 Oktober 2026:** notebook33–35 selesai, GAN fold1 selesai100epoch. Pilot **FAIL** pada9/15 kombinasi kelas–keluarga fitur; notebook36 dihentikan pada sel `require_quality(study, 1)` sebelum fold2. Ini sesuai gate, bukan crash training. [Diagnosis dan langkah perbaikan](ANALISIS_PILOT_GAN_V6_20261004.md). Jangan mengulang36 untuk memaksa lanjut atau mengedit hasilQC; recipe baru perlu diuji pada run terpisah.

**Mulai dari notebook 33, lalu 34 dan 35.** Fase ini menyelesaikan pilot mutu GAN sesuai proposal sebelum eksperimen manfaat classifier. Anda menjalankan training manual di PC. V4/V5 dan notebook 20–32 tetap menjadi hasil yang sudah selesai; tidak perlu dilatih ulang.

Run baru: `20261004_conditional_gan_qc_v6`. Kode baru di `research_v6/`, runner baru `_v6_runner.py`. Output hanya di `results/research_v6/`. Seluruh eksperimen lama dipertahankan.

| Notebook | Jalankan apa | Keluaran utama |
|---|---|---|
| [33 — Persiapan dan smoke](../../notebooks/33_Persiapan_dan_Smoke_GAN_V6.ipynb) | Setup → prepare → smoke MPS → status | Manifest 5 fold, skala amplitudo fit-only, fitur referensi, satu update G/D disposable untuk menguji device |
| [34 — Pilot fold1](../../notebooks/34_Pilot_Training_dan_QC_GAN_Fold1_V6.ipynb) | Train → generate → QC → status | GAN 100 epoch, 64 sintetis per kelas, diagnosis kualitas/memorisasi |
| [35 — Visualisasi](../../notebooks/35_Visualisasi_Mutu_dan_Memorisasi_GAN_V6.ipynb) | Report → tampilkan gambar → status | Loss/exposure, waveform12-lead, spektrum, keluarga fitur gagal, jarak nearest neighbour |
| [36 — Fold2–5](../../notebooks/36_GAN_Fold2_sampai5_Setelah_Pilot_Lulus_V6.ipynb) | Sesudah pilot PASS dan laporannya diperiksa | GAN/QC per fold; berhenti pada screening FAIL |

**Kerjakan 33–35 dahulu, lalu minta audit hasil pilot.** Notebook 36 tidak perlu dijalankan sebelum hasil pilot diperiksa. Jika gate fold1 FAIL, jangan menghabiskan biaya seluruh fold. Laporan tetap tersedia dan dipakai menentukan perbaikan pada run baru.

## Langkah di VS Code/Jupyter

1. Buka folder project ini dan pilih kernel `.venv/bin/python` milik project. Notebook mencari root dari direktori saat ini, sehingga boleh dibuka dari root atau `notebooks/`.
2. Buka 33, jalankan sel satu per satu. Device default `mps`. Prepare hanya audit/cache development; smoke benar-benar menjalankan satu update G dan D yang dibuang. Tidak menyimpan bobot hasil smoke untuk pilot.
3. Jika smoke selesai, buka 34. Sel `train` adalah training data asli. Log epoch tampil langsung, termasuk loss, moment, jumlah draw dan jumlah pasien yang terambil.
4. Setelah training selesai, jalankan `generate` dan `qc`. QC dapat memerlukan waktu karena setiap sintetis dibandingkan dengan seluruh window inner-fit kelas yang sama pada 12×5000 sampel, lalu dibandingkan dengan pasien referensi mutu.
5. Buka 35 meskipun QC FAIL. Periksa `Engineering gate`, `Numeric checks`, `Failed families` dan semua grafik.
6. Sesudah pilot dianalisis dan PASS, 36 dapat menjalankan fold2–5. Setiap sel fold berhenti jika gate FAIL. Classifier belum dilatih pada fase V6.

Jangan menjalankan dua training notebook bersamaan: proses memakai lock yang sama untuk menjaga checkpoint dan penggunaan memori. Tidak ada penurunan batch otomatis.

## Resume dan error

Interrupt sel `train`, tunggu proses berhenti, lalu ulang sel yang sama pada device yang sama. `last.pt` menyimpan model G/D, optimizer, seluruh RNG, RNG dataset/loader/latent, history dan epoch terakhir lengkap. Epoch yang belum lengkap diulang. Hasil lengkap digunakan ulang dengan verifikasi hash. Resume identik telah diuji pada CPU sintetis; determinisme bitwise MPS bukan klaim yang dijamin.

Error ditampilkan lengkap dengan 40 baris terakhir dan log di `manual_logs/`; jadi diagnosis tidak berhenti pada `CalledProcessError`. Jika `Kode/protokol/data/runtime V6 berubah`, jangan mengedit `protocol.json` agar cocok. Pertahankan run lama dan gunakan run-id baru dengan perubahan yang dicatat.

Jika MPS smoke gagal, salin ekor log untuk diagnosis. Pindah ke CPU hanya melalui run-id baru dan device tetap untuk keseluruhan run; tidak melanjutkan checkpoint MPS dengan CPU secara diam-diam.

## Pembagian pasien dan sinyal

Jumlah mengikuti manifest V4 yang dipertahankan, dengan stratifikasi diagnosis dan seed split tetap:

| Fold | GAN fit | Referensi mutu | Outer validation classifier |
|---|---:|---:|---:|
| 1 | 147 | 37 | 46 |
| 2 | 146 | 37 | 47 |
| 3 | 147 | 37 | 46 |
| 4 | 148 | 37 | 45 |
| 5 | 147 | 37 | 46 |

Waveform outer validation classifier terpisah dari fit dan QC GAN pada fold yang sama. **59 pasien test historis** tidak dibaca pada fase V6.

Pasien yang menjadi validation pada fold lain dapat menjadi fit/reference pada fold ini, sebagaimana CV. Semua larangan berlaku di dalam fold yang sama. Referensi mutu berada di dalam outer training; ini bukan label baru atau dataset eksternal. V6 memakai ulang cache, bukan membuat salinan cache besar.

Amplitudo memakai satu skala global: median RMS global pasien GAN-fit, dihitung pada maksimal tiga window per pasien yang dipilih deterministik awal/tengah/akhir. Untuk pasien dengan hanya dua window, keduanya dipakai. Tidak memusatkan/menskala setiap lead secara terpisah, tidak min-max per window, dan tidak memotong amplitudo. Unit tetap unit file/cache; tidak diasumsikan mV.

GAN memakai 2000 Hz dan 5000 sampel = 2,5 detik. Sampling kelas→pasien→window uniform membuat kontribusi pasien seimbang **dalam harapan**, bukan setiap pasien harus terambil sama persis. Anggaran per epoch adalah empat kali jumlah pasien fit, batch16. Riwayat exposure per pasien dan per kelas disimpan.

## Apa arti PASS/FAIL

QC menguji amplitude/roughness/spectrum/correlation/limb residual pada setiap kelas. Distribusi real memberi setiap pasien bobot total satu. Ambang Wasserstein ternormalisasi ditetapkan sebelum training: maksimum 0,5 atau dua kali jarak real-fit ke real-reference. Jika lebih dari 25% fitur dalam suatu keluarga/kelas gagal, screening keseluruhan FAIL. Residual limb lead memakai batas atas, sehingga hubungan sintetis yang lebih tepat secara aljabar tidak dihukum hanya karena DWT cache real meninggalkan residual kecil.

Screening juga menolak nonfinite/lead konstan, exact copy training, near-copy dengan RMSE relatif ≤0,01, duplikasi/near-duplikasi sintetis melebihi 5%. Near-copy dibandingkan full native dengan kelas sama, tanpa phase alignment; bukan sertifikasi privasi atau uji membership inference. Sampel di kelas berbeda juga dibandingkan untuk exact-copy hash terhadap semua inner-fit.

**PASS tidak membuktikan PVC benar, label asal sintetis benar secara klinis, atau classifier meningkat.** Tidak ada ahli ECG dalam proyek. Gambar/fitur numerik hanya mendukung pemeriksaan komputasional. Sinyal asli pasien sulit tidak dibuang atau dilabel ulang. Gate tidak diubah setelah melihat skor untuk memaksa PASS.

## Setelah seluruh GAN lolos

Fase berikut dibuat berdasarkan audit V6: perbandingan **real-only K4 → kontrol augmentasi real → augmentasi GAN** pada lima fold pasien, seed42/43/44 dan recipe classifier yang sama. Kontrol augmentasi diperlukan untuk membedakan manfaat GAN dari bertambahnya keragaman window/training. Scaler classifier tetap fit outer-training; data validation tetap asli dan hasil test historis tidak memilih arm.

Endpoint classifier tersebut **belum tersedia pada V6**. Menunggu pilot mutu adalah urutan eksperimen; tidak ada hasil GAN/classifier baru yang dapat diklaim sekarang. Tiga arsitektur proposal, tuning grid dan metode baru tetap dapat menjadi pembanding setelah pilot memberi bukti mutu. [Rencana penelitian menuju naskah kuat](RISET_DAN_FASE_GAN_V6_20261004.md).

## Output dan kebutuhan penyimpanan

Per fold: `preparation.json`, `amplitude_scale.json`, daftar fit/reference, fitur real, `case.json`, `history.csv`, `last.pt`, `generator.pth`, `discriminator.pth`, `completed.json`, `synthetic.npy`, `generated.json`, `quality.json`, tabel nearest neighbour/fitur, `REPORT.md`, `figures/`.

192 window float32 memerlukan sekitar **44 MiB/fold**, seluruh fold sekitar220 MiB. Checkpoint/optimizer menambah ukuran; estimasi keseluruhan fase sekitar0,5–0,8 GiB, tanpa salinan cache ECG. Disk guard mempertahankan2 GiB bebas ditambah cadangan256 MiB saat menulis output besar. Jangan menghapus hasil lama untuk meningkatkan skor; jika disk kurang, pindahkan arsip terverifikasi secara manual.

Alternatif terminal dari root project:

```bash
.venv/bin/python -m research_v6 prepare
.venv/bin/python -m research_v6 smoke --device mps
.venv/bin/python -m research_v6 train --fold 1 --device mps
.venv/bin/python -m research_v6 generate --fold 1 --device mps
.venv/bin/python -m research_v6 qc --fold 1
.venv/bin/python -m research_v6 report --fold 1
.venv/bin/python -m research_v6 status
```
