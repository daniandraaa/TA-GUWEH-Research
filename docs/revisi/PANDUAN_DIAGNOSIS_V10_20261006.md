# Panduan diagnosis V10 tanpa training ulang

**Langkah berikut adalah notebook 56.** Notebook 51–55 selesai dan 130 kasus V10 valid, tetapi pretrained beku belum melampaui random probe. Diagnosis memisahkan kemungkinan masalah cache/backend dari gap training–validasi sebelum menambah training baru. [Analisis hasil V10](ANALISIS_HASIL_V10_20261006.md).

Buka [56 Diagnosis Transfer V10](../../notebooks/56_Diagnosis_Transfer_V10_Tanpa_Training_Ulang.ipynb), pilih kernel `.venv` project dan jalankan sel berurutan dengan Shift+Enter. Default device **MPS** harus dipertahankan agar reproduksi dinilai pada device yang sama dengan training. Notebook/code V10 lama tetap dibekukan.

| Tahap | Yang dijalankan manual | Bukti selesai |
|---|---|---|
| Setup | Menentukan root, run lama dan device MPS | Path Python/run tampil benar |
| `fits` | Inference enam checkpoint tersimpan: selector inner terbaik dan outer refit fold1/seed42, tiga arm | Probabilitas validation cocok dalam toleransi dan argmax identik; tabel F1 training/validation tersedia |
| `encoder` | Re-encode semua window tiga pasien inner training, masing-masing satu per kelas, pada CPU dan MPS | Patient mean/token cocok dengan cache dalam toleransi; seluruh perbandingan tersedia |
| Ringkasan | Membaca CSV dan JSON diagnostik | Kedua flag reproduksi bernilai `true` atau kegagalan tercatat untuk dianalisis |

**Tidak ada optimizer step, model baru, test atau pemilihan checkpoint baru.** Tahap `fits` menggunakan cached feature/token, bukan mengekstrak seluruh data ulang. Tahap `encoder` membaca ECG asli hanya dari tiga pasien inner training yang dipilih berdasarkan ID pertama per kelas, bukan berdasarkan skor atau tingkat kesulitan. Tidak membuat waveform cache baru.

Hasil tersimpan di:

```text
results/research_v10/20261006_ecg_fm_transfer_pilot_v10/
  diagnostics_20261006_v1/
    fold1_saved_fit_replay_mps.csv
    encoder_cache_replay_mps.csv
    fits_mps_result.json
    encoder_mps_result.json
    fits_mps_completed.json
    encoder_mps_completed.json
```

Log tampil langsung dan disimpan di `manual_logs/`. Tahap encoder memuat ulang bobot resmi besar; tutup aplikasi berat agar RAM cukup. Tidak ada unduhan atau instalasi dependency baru. Jika terhenti sebelum selesai, ulang sel yang sama. Diagnostic selesai dilewati setelah verifikasi identitas/hashes; diagnostic yang hasilnya gagal tetap dilaporkan gagal ketika diulang.

Toleransi reproduksi probabilitas adalah **rtol 1e-4, atol 1e-5**, dengan argmax wajib identik. Toleransi encoder/cache adalah **rtol 1e-3, atol 1e-4**. Toleransi ditetapkan sebelum inference; jika gagal, angka error dan sumber perbandingan harus diperiksa, bukan menaikkan ambang untuk meloloskan hasil. Tiga pasien merupakan pemeriksaan terbatas dan tidak membuktikan kesetaraan backend pada semua pasien.

Metrik training yang baru dihitung berasal dari checkpoint terpilih dalam mode eval; metrik tersebut lebih sebanding dengan validation daripada akurasi online selama optimizer update. Jika reproduksi lolos tetapi gap training–validasi besar, tahap perbaikan berikut berfokus regularisasi head dan kapasitas. Jika reproduksi berbeda, penyebab cache/device diperiksa dahulu. Jika keduanya baik, lanjut ablation input/durasi/pooling dan sumber pretrained dengan kontrol yang sama. Semua skor diagnosis bersifat posthoc dan tidak boleh mengganti pilihan model berdasarkan outer fold.

**Setelah notebook 56 selesai, kirim hasilnya. Tidak perlu mengulang 51–55 atau melakukan final/test baru.** Prototype diagnosis sudah lulus lima test CPU sintetis; inference ECG asli notebook ini belum dijalankan Codex.
