# Reproduksibilitas: audit dahulu, bukan full run siap pakai

## 1. Akses dan baca hasil tanpa training
Akun yang diberi akses private dapat clone `https://github.com/daniandraaa/TA-GUWEH-Research.git`. Mulai dari hasil_eksperimen/ dan protocol YAML. Ini tidak membutuhkan dataset/GPU. Link README_PROJECT_ASLI.md yang absolut ke Mac adalah historis.

## 2. Ambil input model hanya jika berizin
```bash
gh release download supervisor-discussion-2026-10 --repo daniandraaa/TA-GUWEH-Research --pattern 'TA_GUWEH_Project_Sharing.zip*' --dir ./private_download
cd private_download
sha256sum -c TA_GUWEH_Project_Sharing.zip.sha256
```
Checksum harus dc6f5e70a51174cb2c597f3c5cb6ba797b77d21ec81a95761c722291766cf19a. Extract ZIP ke folder terpisah; root di dalamnya TA GUWEH/. Jangan commit/push arrays setelah extract. Canonical input berada di results/research_v2/20260930_anatomy_fs2000_v2/data/{development,test}/X.npy dan segments.csv. Baca PACKAGING_DATASET_DICTIONARY.md dan verifikasi fold/labels/order. Jangan membuat split baru tanpa mendokumentasikan perubahan.

## 3. Environment dan dependensi
PACKAGING_DEPENDENCIES.txt adalah inventory 134 distribution Mac dari METADATA, bukan portable lock teruji. Buat environment terisolasi yang kompatibel dengan torch/numpy/pandas/sklearn/scipy/yaml dan dependency proyek; jangan install daftar Mac membabi-buta di Linux. Periksa import dan tests sebelum eksperimen. Tidak ada environment project yang dipasang atau tests training yang dijalankan untuk publikasi ini. CLI V13/V14 menawarkan cpu/mps, bukan CUDA flag; MPS hanya pada Mac kompatibel.

## 4. Adaptasi untuk eksperimen baru (belum dijalankan)
Baca research_v13/common.py, research_v14/common.py dan research_v14_final/run.py: context/hash membekukan source/predecessor, data, folds, kode, runtime; final V14 memeriksa hash preregistrasi. Metadata anonim dan beberapa artefak yang hilang membuat run ID lama tidak kompatibel. **Jangan mematikan pemeriksaan hash agar terlihat berhasil atau menimpa hasil historis.** Restore hanya artefak berizin yang kurang atau buat pipeline/snapshot/run ID baru dengan provenance dan audit split/scaler/GAN eksplisit. V14 final mempunyai output path tetap; adaptasinya membutuhkan review dan preregistrasi baru, bukan sekadar --run-id.

Sesudah adaptasi, antarmuka source V13 adalah `python -m research_v13 <stage> --device cpu --run-id <run-baru>`; stage prepare/smoke/gan-train/gan-generate/gan-qc/cv/select/final/test/analyze/shap/status. V14 CV memakai `python -m research_v14 <stage> --device cpu --run-id <run-baru>`, stage prepare/smoke/cv/compare/status. Ini dokumentasi antarmuka kode, **bukan perintah yang dijamin sukses pada paket sekarang**. Review urutan dan source run protocol sebelum eksekusi, reserve test baru jika mengubah desain, jangan retuning historical test.

Kriteria reproduksi baru: audit grouping pasien/overlap, scaler dan GAN fit hanya training fold, lingkungan terkunci, artifacts/hash baru, semua seed/fold, endpoint konsisten, agregasi probabilitas ensemble, dan hasil baru disimpan terpisah. Laporkan perbedaan terhadap hasil historis, jangan mengklaim checksum anonim membuktikan identitas eksperimen lama.
