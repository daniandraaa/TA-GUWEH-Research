# Perbaikan notebook 22: adaptive pooling Lead–Time pada MPS

Error berasal dari sel `runner.run_stage("pilot", "--model", "lead_time")`, pada log `20261002T122258_294545Z_22_v4_pilot.log`. Pesan utamanya: `Adaptive pool MPS: input sizes must be divisible by output sizes`.

CNN mengubah panjang input 5.000 menjadi 157 posisi fitur. `AdaptiveAvgPool1d(16)` dapat memprosesnya di CPU, tetapi kernel MPS yang terpasang menolak panjang yang tidak habis dibagi 16. Ini sesuai [isu resmi PyTorch #96056](https://github.com/pytorch/pytorch/issues/96056). Kasus pertama Lead–Time baru mempunyai config; belum ada epoch, checkpoint atau metrics. Diagnosis berjumlah 45/45, pilot empat model lainnya berjumlah 40/50. Hasil yang sudah selesai dipertahankan.

Modul `research_v4_mps_compat/` mengganti hanya pooling terakhir Lead–Time dengan operasi slice, mean dan stack pada perangkat yang sama. Batas bin mengikuti [implementasi ATen](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/AdaptivePooling.h): awal `floor(i*N/K)` dan akhir eksklusif `ceil((i+1)*N/K)`. Seluruh posisi, termasuk ekor dan overlap bin, tetap dihitung. Parameter dan key checkpoint sama. Penjumlahan floating point dapat menghasilkan perbedaan pembulatan kecil; patch bukan perubahan jumlah bin atau hipotesis model.

Kode `research_v4/`, protokol, config kasus, fold, learning rate, batch dan seed tetap sesuai fingerprint run asli. Patch tidak disembunyikan dalam fingerprint lama: `backend_compatibility.json` mencatat hash seluruh kode patch, fingerprint eksperimen asal serta hash config/metrics yang sudah ada saat aktivasi. Setiap fit Lead–Time juga mempunyai sidecar tersebut. Perubahan patch kemudian ditolak saat resume atau test. Hasil Lead–Time lama tanpa provenance patch tidak boleh dicampur otomatis. File laporan verifikasi ada di `reports/backend_compatibility_validation.json` pada run ini.

Verifikasi pada 2 Oktober 2026:

- 21 uji CPU V4 lulus: nilai/gradien pooling float32 dan float64, panjang 157→16, kasus divisible dan input lebih pendek dari jumlah bin, tail, checkpoint/parameter, output dan gradien model native, serta audit perubahan patch.
- Uji langsung MPS dengan PyTorch 2.12.0 lulus: kesetaraan nilai/gradien pooling, Lead–Time mean dan attention pada input 5.000 sampel, backward, satu langkah AdamW, dan inference semua segment secara chunk.
- Uji GPU memakai tensor acak, dua pasien dengan dua segment. Kelayakan minibatch nyata tetap diperiksa oleh pilot manual; uji singkat ini tidak menilai skor klasifikasi.
- Notebook yang sudah dieksekusi beserta output error dipertahankan; tidak ada training dataset atau pembacaan historical test oleh Codex dalam perbaikan ini.

Untuk melanjutkan di notebook 22:

1. Jalankan kembali **sel setup**, termasuk `importlib.reload(_v4_runner)` dan pembuatan `runner`.
2. Jalankan sel **Lead–Time**: `runner.run_stage("pilot", "--model", "lead_time")`.
3. Setelah sepuluh kasus Lead–Time selesai, jalankan sel `report`; pilot lengkap menjadi 50 kasus. Lanjutkan full grid 23–25 sebelum seleksi/final pada 26.

Runner otomatis menjalankan entrypoint baru. Terminal dapat memakai:

```bash
.venv/bin/python -m research_v4_mps_compat pilot --model lead_time --device mps
```

Gunakan entrypoint kompatibilitas yang sama untuk CV, stability, final dan test. Memanggil `python -m research_v4` langsung tetap memakai implementasi pooling lama. Menjalankan ulang empat sel pilot yang selesai akan melewati kasus lengkap, tetapi tidak diperlukan untuk memperbaiki error ini.
