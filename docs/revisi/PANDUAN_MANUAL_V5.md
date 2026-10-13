# Panduan manual V5 — ablation jumlah window training

**Status 4 Oktober 2026: notebook 29–32 selesai, kedua smoke MPS dan 30/30 training ECG asli lulus eksekusi.** Audit tambahan cocok dengan prediksi/laporan; K8 belum memenuhi kriteria sehingga baseline K4 dipertahankan. [Analisis hasil V5](ANALISIS_HASIL_V5_20261004.md). Alur di bawah menjadi referensi yang sudah dijalankan; tidak perlu training ulang. Notebook dengan output pengguna dan hasil V4/V5 dipertahankan.

Tujuan tahap ini adalah menguji maksimal **4 versus 8 window training per pasien**. Batch tetap **8 pasien**. Hierarchical menggunakan pooling mean/LR 5e-4 dan Encoder attention/LR 1e-4 dari CV V4. Kapasitas maksimal segmen per update bertambah dari 32 menjadi 64 sebagai konsekuensi jumlah window bertambah. Preprocessing, pasien/fold, scaler, class weighting, model, optimizer, scheduler dan aturan early stopping mengikuti recipe selected V4. Validation tetap memakai seluruh window tiap pasien.

## Urutan menjalankan

| Urutan | Notebook | Yang dijalankan | Tanda selesai |
|---|---|---|---|
| 1 | [29 — Persiapan](../../notebooks/29_Persiapan_Ablation_Window_V5.ipynb) | Setup → prepare → konfigurasi → smoke dua model → status | Kedua model `SMOKE PASS`; training baru 0/30 |
| 2 | [30 — Hierarchical](../../notebooks/30_Ablation_Window_Hierarchical_V5.ipynb) | Setup → pilot fold1 seed42 → lihat pilot → full CV → status | `per_model.hierarchical: 15` |
| 3 | [31 — Encoder](../../notebooks/31_Ablation_Window_Encoder_V5.ipynb) | Setup → pilot fold1 seed42 → lihat pilot → full CV → status | `per_model.encoder: 15`, total 30/30 |
| 4 | [32 — Analisis](../../notebooks/32_Analisis_Ablation_Window_V5.ipynb) | Setup → status → report → tabel → grafik | Perbandingan lengkap development tersedia |

Pilih kernel Python `.venv` project dan gunakan **Shift+Enter** satu sel demi satu sel. Tunggu sel training selesai sebelum lanjut. Jangan menjalankan notebook 30 dan 31 secara bersamaan. Runner selalu memakai interpreter `.venv/bin/python`, mencetak epoch/progres langsung, dan menyimpan log beserta bagian akhir traceback.

Smoke pada notebook 29 memakai delapan pasien **training fold1** yang masing-masing mempunyai sedikitnya delapan window. Satu update optimizer dilakukan untuk memeriksa kapasitas 64 segmen dan nilai finite. Bobotnya dibuang; smoke bukan hasil CV atau model final. **Kedua smoke MPS sudah dijalankan pengguna dan lulus**, diikuti 30 fit lengkap. Pengujian persiapan oleh Codex sebelumnya menggunakan CPU/sinyal sintetis; audit terbaru memeriksa artefak eksekusi nyata tanpa training tambahan.

Training CV memakai maksimal 100 epoch dengan patience 15, mengikuti V4. Selesai sebelum 100 epoch adalah perilaku early stopping yang normal. Pilot masing-masing model adalah fold1 seed42 dalam desain yang sama; setelah selesai ia dilewati oleh full CV. **Total hanya 30 fit baru**, yaitu dua model × lima fold × tiga seed (42, 43, 44). Sebanyak 30 hasil kontrol development V4 digunakan lewat manifest asal/hash, bukan dilatih ulang. Folder output baru:

```text
results/research_v5/20261003_window_bag_ablation_v5/
  protocol.json                 # identitas run terpisah
  control_manifest.json         # rujukan/hash 30 kontrol V4
  scalers/fold_1 ... fold_5/     # salinan scaler training fold V4
  smoke/                        # pemeriksaan perangkat manual
  ablation/hierarchical/        # 15 fit baru
  ablation/encoder/             # 15 fit baru
  manual_logs/                  # progres dan traceback
  reports/                      # perbandingan setelah 30/30 selesai
```

## Jika berhenti atau error

Ulang **sel yang sama** untuk resume dari epoch terakhir yang selesai. Kasus yang selesai dilewati. Gunakan device `mps` yang sama dengan kontrol V4 dan jangan mengganti LR, batch, seed, jumlah window atau kode setelah run dimulai. Perubahan identitas run ditolak supaya hasil berbeda tidak tercampur.

Jika smoke/training mengalami kehabisan memori, tutup proses GPU lain lalu ulang sel. Program tidak menurunkan batch otomatis. Bila masih gagal, simpan traceback/log untuk menentukan desain baru; mengganti batch pasien menjadi empat akan mengubah faktor lain dalam ablation. Jika disk bebas kurang dari batas yang ditetapkan, endpoint berhenti dan menampilkan kebutuhan ruang; tidak ada file hasil lama yang dihapus otomatis.

Perbandingan notebook 32 terkunci hingga seluruh 30 fit lengkap. Jangan memilih metode hanya dari pilot, seed terbaik, atau hasil test historis. Tahap V5 ini tidak menyediakan endpoint final/test dan tidak menghasilkan skor test baru.

## Cara membaca hasil

`model_summary.csv` dan `seed_summary.csv` menunjukkan K4/K8, perubahan mean macro-F1 fold, recall tiap kelompok, variasi seed, collapse dan biaya komputasi. `folds_paired.csv` mencatat semua pasangan fold/seed. Pooled OOF memakai seluruh 230 pasien development setiap seed; nilainya dibedakan dari mean lima fold. Grafik tersedia dalam PNG dan PDF.

Aturan prioritas ditetapkan sebelum training: mean perubahan macro-F1 fold positif, peningkatan pada sedikitnya dua dari tiga seed, dan rata-rata recall kedua kelompok RVOT tidak turun. Baca juga recall masing-masing kelas dan collapse karena nilai rata-rata dapat menutupi trade-off. Status prioritas hanya menentukan eksperimen development berikut; belum mengganti model primer atau menjadi bukti generalisasi. Fold/seed tidak dianggap 15 pengamatan independen. Bila K8 tidak memperbaiki hasil, pertahankan kontrol dan gunakan hasil negatif itu untuk memilih hipotesis selanjutnya.

Setelah notebook 32 selesai, review tabel/variasi seed untuk menentukan kebutuhan tuning ulang atau perubahan representasi. GAN terkoreksi bila tetap dalam objektif proposal, penguatan evaluasi dan Bab III–V masih pekerjaan berikutnya. Target macro-F1 0,9 belum terbukti bisa dicapai dari ablation ini.

Alternatif terminal dari folder project, jalankan berurutan:

```bash
.venv/bin/python -m research_v5 prepare
.venv/bin/python -m research_v5 smoke
.venv/bin/python -m research_v5 pilot --model hierarchical
.venv/bin/python -m research_v5 cv --model hierarchical
.venv/bin/python -m research_v5 pilot --model encoder
.venv/bin/python -m research_v5 cv --model encoder
.venv/bin/python -m research_v5 status
.venv/bin/python -m research_v5 report
```

Jika ingin hanya satu kasus untuk penelusuran setelah pilot selesai:

```bash
.venv/bin/python -m research_v5 cv --model hierarchical --fold 2 --seed 43
```

[Dasar audit input](ANALISIS_AUDIT_INPUT_V4_20261003.md) · [Fase penyelesaian](FASE_SETELAH_V4_20261003.md).
