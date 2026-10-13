# Fase classifier setelah GAN V7 lolos — 5 Oktober 2026

**Pembaruan 6 Oktober 2026: V8/notebook 41–45 selesai, 120 fit valid; manfaat GAN belum konsisten.** [Analisis hasil](ANALISIS_HASIL_V8_20261006.md) · [fase setelah V8](FASE_SETELAH_V8_20261006.md). Rancangan dan implementasi di bawah dipertahankan sebagai jejak fase yang telah dijalankan.

**GAN/QC V7 selesai dan terverifikasi pada lima fold. Pertanyaan berikut adalah apakah augmentasi membantu lokalisasi pasien melebihi real-only serta augmentasi real biasa.** [Hasil V7](ANALISIS_HASIL_V7_20261005.md).

**Implementasi V8 kini tersedia pada notebook 41–45.** Protokol telah menetapkan satu penggantian pada bag K4 penuh, sumber augmentasi, selector epoch inner real-only, serta anggaran sama untuk ketiga arm. [Panduan eksekusi V8](PANDUAN_MANUAL_V8.md). Keputusan rinci di bawah merupakan rancangan sebelum implementasi; pilihan aktual dijelaskan dalam panduan. Jangan menjalankan ulang notebook 37–40 untuk fase ini. Implementasi baru harus memakai paket/run baru, mempertahankan kode dan hasil V4–V7.

## Pembanding pertama

| Arm | Tujuan | Kontrol yang perlu dipatok |
|---|---|---|
| K4 real-only | Baseline classifier | Fold pasien, model/pooling, seed, preprocessing/scaler, sampling, update budget |
| Kontrol augmentasi real | Memisahkan efek mekanisme augmentasi dari sintesis GAN | Peluang/porsi augmentasi serta mekanisme perubahan bag sama dengan arm GAN; sumber real training saja |
| Real+GAN V7 | Menguji manfaat sinyal sintetis yang telah lolos QC | Generator sesuai fold, sampling kondisi diagnosis, porsi sintetis tetap, validation seluruhnya real |

Prioritas awal **Hierarchical dan Encoder**, dua kandidat CV V4 teratas; lima fold × tiga seed classifier × tiga arm × dua model = **90 kasus pembanding**, ditambah **30 selector epoch inner**; total 120 fit. Pilot delapan fit termasuk anggaran itu. Semua kontrol classifier dilatih baru karena epoch selection V8 berbeda. Penggunaan ulang real-only V4/V5 hanya sah bila seluruh recipe, anggaran, checkpoint selection, scaler dan makna batch sama; bila ada perbedaan, train kontrol pada run baru. PatchTST tetap kandidat perluasan untuk memenuhi perbandingan tiga arsitektur proposal bila tahap awal menjanjikan.

## Rancangan sebelum implementasi — pilihan aktual ada pada panduan V8

1. Tetapkan cara memasukkan sintetis pada model MIL K4. Bag sintetis atau penggantian window tidak mempunyai identitas pasien nyata; semua arm harus mencatat sumber/mekanisme dengan jelas.
2. Tentukan porsi augmentasi pertama, misalnya satu dari empat window per bag sebagai hipotesis awal, lalu buat kontrol real yang menggunakan mekanisme setara. Nilai ini belum dipilih sebagai protokol final. Catat exposure real/sintetis dan peluang penggunaan ulang 64 window per kelas, tanpa menambah update hanya pada arm GAN.
3. Tetapkan pemilihan epoch dan anggaran yang sebanding. Jika memakai inner validation, pemilihan checkpoint dilakukan di inner dan outer pasien hanya dinilai setelah training; generator/scaler classifier tidak menggunakan outer waveform. Reuse recipe CV V4 harus disebut eksploratif karena development sebelumnya sudah ikut pemilihan recipe.
4. Setiap fold hanya menggunakan generator V7 fold tersebut. Pasien referensi mutu dapat memiliki peran classifier tersendiri yang dicatat, tetapi tidak boleh diselundupkan ke data GAN-fit. Manifest asal dan hash target/checkpoint/sintetis harus terikat case classifier.
5. Buat smoke perangkat disposable, pilot satu fold untuk semua arm, kemudian full comparison secara manual. Resume menyimpan seluruh RNG sampling augmentasi; hindari mengubah generator atau rasio di tengah run.

## Hasil yang menjadi dasar keputusan

Macro-F1 pasien dan recall kedua RVOT pada validation real merupakan keluaran utama. Laporkan perbandingan arm pada fold/seed yang sama, pooled OOF, arah efek per seed, collapse/kegagalan, variasi, exposure dan biaya. Jangan memperlakukan 15 hasil fold/seed sebagai pasien independen atau memilih hanya seed yang menang. Jangan memakai skor test historis untuk memilih rasio, arm atau seed.

Jika manfaat konsisten, lanjutkan ablation objective/lead dan seed GAN tambahan, serta evaluasi pasien yang lebih independen dan penulisan. Jika manfaat kecil atau negatif, hasil QC tetap berguna sebagai hasil metodologis, tetapi tidak diklaim meningkatkan lokalisasi; prioritaskan alternatif representasi atau pretraining dengan kontrol setara.

Skor classifier baru belum tersedia. PASS mutu GAN dan 960 window sintetis tidak menjamin macro-F1 0,9, validitas klinis, atau penerimaan jurnal.
