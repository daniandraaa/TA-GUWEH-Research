# Menjalankan fase V4 secara manual

**Status 3 Oktober 2026: seluruh notebook 20–27 telah selesai.** Diagnosis 45/45, CV 450/450, stability 75/75 (25 reuse CV), final 25/25 dan test 25/25 telah diaudit. [Analisis hasil dan tindak lanjut](ANALISIS_HASIL_V4_20261003.md). Gunakan notebook 27 untuk melihat laporan; alur training di bawah menjadi referensi, tidak perlu diulang untuk run yang sudah selesai.

**Riwayat perbaikan notebook 22 (sudah selesai dijalankan):** jalankan ulang sel setup yang berisi `importlib.reload(_v4_runner)` dan pembuatan `runner`, kemudian hanya sel `runner.run_stage("pilot", "--model", "lead_time")`, lalu sel `report`. Empat model sebelumnya sudah selesai; tidak perlu diulang. Runner kini memakai `research_v4_mps_compat` secara otomatis. Rincian penyebab, verifikasi dan provenance tersedia di [catatan perbaikan MPS](PERBAIKAN_MPS_LEAD_TIME_V4_20261002.md).

Verifikasi patch: 21 uji khusus V4 lulus pada CPU, termasuk delapan uji kompatibilitas. Uji MPS sintetis dengan input 5.000 sampel, mean/attention pooling, forward/backward, satu langkah AdamW dan inference juga lulus. Output/gradien pooling cocok dengan referensi CPU dalam toleransi floating point. Verifikasi Codex tidak menjalankan training pada dataset atau membuka sinyal historical test. Persiapan/scaler telah diperiksa pada 230 pasien/2.274 window development. Estimasi bobot final seluruh model/seeds sekitar 151 MiB, di luar optimizer aktif, log/prediksi dan overhead file.

| Urutan | Notebook | Pekerjaan |
|---|---|---|
| 20 | [Persiapan protokol](../../notebooks/20_Persiapan_Protokol_V4.ipynb) | Validasi cache/fold/scaler; tanpa training/test signal |
| 21 | [Diagnosis native V3](../../notebooks/21_Diagnosis_Stabilitas_Native_V3.ipynb) | Tiga sel model, masing-masing 15 fit development |
| 22 | [Pilot](../../notebooks/22_Pilot_Model_dan_Pooling_V4.ipynb) | Lima sel model, masing-masing 10 fit S04 |
| 23 | [Grid tiga pembanding](../../notebooks/23_Grid_Pembanding_Encoder_Patch_Hierarchical_V4.ipynb) | Encoder, patch, Hierarchical; 270 kasus termasuk pilot |
| 24 | [Residual multiscale](../../notebooks/24_Grid_Residual_Multiscale_CNN_Transformer_V4.ipynb) | 90 kasus termasuk pilot |
| 25 | [Lead/time transformer](../../notebooks/25_Grid_Lead_Time_CNN_Transformer_V4.ipynb) | 90 kasus termasuk pilot |
| 26 | [Seleksi, stability, final/test](../../notebooks/26_Seleksi_Stabilitas_Final_Test_V4.ipynb) | Setelah 45 diagnosis dan 450 CV lengkap; 50 fit stability tambahan, 25 final, kemudian inference test |
| 27 | [Visualisasi proses](../../notebooks/27_Visualisasi_Proses_dan_Hasil_V4.ipynb) | Boleh diulang setelah 20; tahap belum selesai tampil pending |

Di setiap notebook, jalankan sel setup dahulu lalu sel endpoint satu per satu (Shift+Enter). Tidak ada flag `False` yang membuat tahap training terlewati diam-diam. Pilot bukan full grid; notebook 26 menolak final bila grid belum lengkap. Jalankan satu proses MPS pada satu waktu. Saat notebook 21 selesai, notebook 27 dapat dipakai memeriksa tabel collapse/stabilitas sebelum pilot.

Untuk berhenti, gunakan Interrupt/Stop. Ulang sel yang sama pada perangkat yang sama untuk melanjutkan akhir epoch tersimpan; kasus selesai dilewati. Jangan mengubah `research_v4/*.py`, `research_v4_mps_compat/*.py`, protocol.yaml, runtime/dependensi atau cache saat run belum selesai. Patch mempunyai fingerprint tambahan yang diperiksa pada setiap fit Lead–Time dan sebelum inference test. Log mencantumkan error sebenarnya, tidak hanya CalledProcessError.

Run default: `20261002_patient_mil_hybrids_v4`. Hasil/log tersimpan di `results/research_v4/<RUN_ID>/`; prediksi dan log per epoch dalam `diagnostic`, `cv`, `stability`, `final`, `test`; tabel/grafik dalam `reports`. Semua notebook harus memakai run-id sama.

**Satuan batch V4:** grid 16/32/64 merupakan maksimum segment per minibatch; maksimal empat segment per pasien membuat patient minibatch 4/8/16. Evaluasi menggunakan seluruh segment pasien. Detail formula, perubahan recipe dan batas interpretasi ada pada [kajian V4](RISET_METODE_DAN_PROTOKOL_V4_20261002.md).

Alternatif terminal, baris dijalankan satu per satu dari folder project:

```bash
cd "/Users/daniandraaa/Desktop/TA GUWEH"
.venv/bin/python -m research_v4_mps_compat prepare --device cpu
.venv/bin/python -m research_v4_mps_compat diagnostic --model encoder --device mps
.venv/bin/python -m research_v4_mps_compat diagnostic --model patchtst --device mps
.venv/bin/python -m research_v4_mps_compat diagnostic --model hierarchical --device mps
.venv/bin/python -m research_v4_mps_compat pilot --device mps
.venv/bin/python -m research_v4_mps_compat cv --model encoder --device mps
.venv/bin/python -m research_v4_mps_compat cv --model patchtst --device mps
.venv/bin/python -m research_v4_mps_compat cv --model hierarchical --device mps
.venv/bin/python -m research_v4_mps_compat cv --model residual_multiscale --device mps
.venv/bin/python -m research_v4_mps_compat cv --model lead_time --device mps
.venv/bin/python -m research_v4_mps_compat select --device cpu
.venv/bin/python -m research_v4_mps_compat stability --device mps
.venv/bin/python -m research_v4_mps_compat final --device mps
.venv/bin/python -m research_v4_mps_compat test --device mps
.venv/bin/python -m research_v4_mps_compat report --device cpu
```

`status` menunjukkan jumlah CV lengkap; `report` membuat laporan kapan pun sesudah preparation. `--fold 1` dan `--scenario S04` tersedia untuk menjalankan sebagian grid secara eksplisit; ini tidak mengurangi kewajiban grid penuh untuk seleksi. `--recipe AdamW` hanya untuk ablation diagnosis opsional, bukan pengganti 45 diagnosis Adam default. Jika disk guard/memori gagal, error menyebut penyebab dan tidak mengubah batch atau menghapus hasil lama.

Target 0,9 belum tercapai; hasil V4 sudah diukur dan diaudit lengkap. Skor utama adalah macro-F1 pasien tiga kelompok. Akurasi/AUC dilaporkan terpisah, semua seed tetap ditampilkan, dan 59 pasien test tetap hold-out historis yang pernah diamati. GAN terkoreksi, nested CV dan validasi eksternal anatomi masih fase lanjutan; notebook ini tidak menyatakan penelitian sudah cukup diterima Q1.
