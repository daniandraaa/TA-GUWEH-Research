# Perbaikan notebook 58 — 7 Oktober 2026

## Penyebab error yang tersimpan

Log `20261006T100736_655832Z_ludb.log` menunjukkan **200/200 rekaman telah diunduh**, lalu benchmark berhenti pada guard disk: ruang 3,02 GiB lebih kecil dari kebutuhan cadangan 3 GiB + 24 MiB. Angka yang ditampilkan dibulatkan; selisih kecil tersebut tetap dapat menyebabkan guard berhenti. Benchmark belum mulai.

Dua log berikutnya, `20261006T171642_994239Z_ludb.log` dan `20261006T171745_275588Z_ludb.log`, menunjukkan `FileNotFoundError` untuk `completed.json` cache fitur V10 di `~/Library/Caches/TA-GUWEH-ECGFM/.../features/`. Folder cache induk tidak tersedia saat pemeriksaan; penyebab hilangnya tidak dapat dipastikan dari log. Source awal memverifikasi cache encoder pada semua endpoint, sehingga LUDB ikut terblokir walau tidak memerlukan embedding. Ruang kini sekitar **4,19 GiB** pada pemeriksaan, tetapi cache tetap tidak tersedia.

## Perbaikan yang tersedia

Source `research_v11/` dan runner `_v11_runner.py` tetap dibekukan. Paket `research_v11_recovery/` dan `_v11_recovery_runner.py` menambahkan perbaikan dependency serta parsing anotasi. Run baru:

`20261007_morphology_regularized_features_v11_r1`.

Protokol baru mengikat hash source asal, source perbaikan, runtime, fold, label, cohort dan audit V10. Source/protokol/log/hasil V10 serta run V11 awal dipertahankan. Byte copy notebook 57–61 sebelum perubahan, termasuk error/output notebook 58, ada di `arsip/notebooks_sebelum_perbaikan_cache_v11_20261007/` dengan manifest hash.

- `prepare/smoke/ludb/extract/qa/analyze/status` tidak memerlukan cache encoder.
- `qrs_rr/morphology/morphology_rr` tetap memerlukan LUDB PASS dan fitur morfologi lengkap, tetapi tidak membuka embedding yang tidak dipakai.
- `random_l2/pretrained_l2/fusion_random/fusion_pretrained` memerlukan file embedding terkait dan `patients.csv` yang **bit-identik dengan hash audit V10**. Tidak ada array pengganti, regenerasi otomatis atau perubahan label.
- Notebook 60/61 menandai arm encoder/fusion yang belum tersedia sebagai **TERTUNDA**; hasil sebagian tidak disebut full comparison tujuh arm. Tiga arm saja berjumlah 150 fit termasuk pilot, sementara seluruh rencana tetap 350 fit.

Jika ada backup fitur, environment `TA_V11_FEATURE_CACHE` dapat menunjuk folder salinannya. Tiga file yang dibutuhkan seluruh encoder arm adalah `patients.csv`, `random_probe.npy`, dan `pretrained_probe.npy`; masing-masing dibandingkan dengan digest yang tersimpan pada audit V10. Manifest cache penuh/prelast tokens tidak dibutuhkan untuk head/fusion yang hanya memakai tiga file itu. Salinan yang berbeda ditolak. Jika tidak ada backup, regenerasi encoder harus direncanakan dan dijalankan manual; perbaikan ini tidak menjalankannya.

## Masalah parser yang ditemukan saat pemeriksaan

Pembacaan **anotasi saja**, tanpa membaca waveform atau menjalankan detektor, menemukan **21.965 puncak QRS**, dengan **21.843 referensi batas lengkap** dan **122 batas hilang/tidak konsisten pada 27 pasangan rekaman–lead**. Contoh pertama rekaman 7/V1: peak tanpa onset sebelumnya, sehingga parser yang mensyaratkan triplet berurutan akan berhenti. Jumlah di sini adalah yang terbaca dari berkas lokal versi 1.0.1, bukan angka yang dipaksakan dari ringkasan publikasi.

Parser perbaikan mempertahankan semua puncak beranotasi untuk evaluasi deteksi. Ia hanya mengambil batas yang langsung bersebelahan dengan anotasi beat; tidak meminjam batas P/T, mengurutkan ulang anotasi yang kontradiktif atau menjadikan peak sebagai onset. Bila ground truth batas tidak lengkap/urutannya salah, galat onset/offset/durasi tidak dihitung untuk beat itu. Beat tersebut **tetap masuk denominator coverage batas**, sehingga missing reference tidak dapat menaikkan coverage secara artifisial. Ketersediaan referensi dan kegagalan estimasi dicatat terpisah dalam CSV.

Ambang detection F1/coverage/galat dan algoritme delineation V11 **tidak diubah**. Sesudah benchmark manual, `ludb/annotation_audit/annotation_integrity.json` menjelaskan masalah referensi dan terikat pada hash manifest benchmark. PASS tetap gate rekayasa, bukan validasi klinis PVC/origin.

## Cara melanjutkan

1. **Restart Kernel**, buka notebook **57**, lalu jalankan dari sel pertama. Setup harus menampilkan run-id dengan akhiran `_r1`.
2. Buka notebook **58** yang telah diperbaiki dan jalankan dari sel pertama. `DOWNLOAD_LUDB=False` menggunakan dataset lokal yang telah diunduh. Ubah menjadi True hanya jika ada file yang memang belum lengkap; file resmi tetap diverifikasi hash.
3. Periksa `summary.json` dan laporan integritas anotasi. Bila gate FAIL, simpan diagnosis; tidak melanjutkan training arm morfologi.
4. Sesudah pengukuran layak, lanjut 59 untuk ekstraksi/QA. Cache encoder yang hilang masih perlu dipulihkan sebelum baseline encoder/fusion dapat dibandingkan.

**Validasi perbaikan:** 19 tes sintetis CPU lulus, mencakup missing cache, backup subset/hash mismatch, gate tetap wajib, mempertahankan anotasi tidak lengkap dalam denominator, serta pipeline/audit/resume V11 sebelumnya. Kelima notebook diperiksa struktur dan sintaksnya. Pemeriksaan anotasi bukan benchmark waveform. Belum ada training ECG asli, ekstraksi lokal, atau hasil deteksi/durasi LUDB baru oleh Codex.
