# Revisi V2 — anatomi, sampling, dan ablation terkontrol

> **Hasil aktual 1 Oktober 2026:** seluruh 270 CV, 15 model final, dan 15 evaluasi test V2 telah selesai. Encoder tetap model utama dari CV; test macro-F1 seed 42 = 0,635, mean lima seed = 0,616 ± 0,034. Analisis menemukan 25 pasien development salah pada ketiga model serta kebingungan antara kelompok RVOT. Lihat [analisis lengkap](ANALISIS_HASIL_V2_20261001.md) dan notebook `10_Analisis_Hasil_V2.ipynb`. Prioritas berikutnya adalah review/kurasi PVC, kontribusi training antar-pasien, dan pengujian terkontrol; GAN baru/validasi eksternal belum selesai.


Status 30 September 2026: kode dan notebook disiapkan untuk dijalankan manual. Belum ada hasil training V2. Pengguna menyetujui tiga kelompok dipertahankan dengan koreksi nama anatomi.

## Koreksi yang mendahului training

Paper yang diberikan pengguna: [Zheng et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8962834/). Descriptor dataset publik: [Zheng et al., Scientific Data 2020](https://www.nature.com/articles/s41597-020-0440-8). Descriptor menyebut 334 pasien, 77 Left/257 Right, 12 sadapan dan 2.000 Hz; struktur dan jumlah ini cocok dengan file lokal. Kohort paper 2022 berjumlah 545 pasien: jangan menuliskan bahwa seluruhnya tersedia di project. Checksum arsip unduhan asli belum diverifikasi.

Kedua sumber menempatkan LC/AC/RC di RVOT. Validasi asal label melalui ablasi pada dataset mendukung dasar label sumber, tetapi tidak otomatis memvalidasi pengelompokan lokal, sinyal sintetis, preprocessing, atau model baru.

| Label numerik tetap | Nama historis | Nama V2 | Sublokasi lokal |
|---|---|---|---|
| 0 | LVOT | LVOT | LCC, RCC, AMC, LCC-RCC Ommisure |
| 1 | Non-OT | RVOT_cusp | LC, AC, RC |
| 2 | RVOT | RVOT_septal_freewall_other | PosteriorSeptal, AnteriorSeptal, FreeWall, RVOTOther |

Tiga kelas ini adalah kelompok anatomi khusus studi. Kelompok ketiga mencakup RVOTOther yang tidak boleh dianggap pasti non-cusp. Sebanyak 289 pasien dan label numeriknya tetap; Summit/unknown tetap dikecualikan untuk menjaga kohort, bukan karena seluruh Summit pasti di luar LVOT. Perlu jelaskan kriteria eksklusi tersebut di naskah. Skor lama tetap dapat ditelusuri, tetapi klaim deteksi Non-OT harus dicabut/dikoreksi.

**Sampling:** 5.000 sampel berarti 2,5 detik pada 2.000 Hz. HPF lama menggunakan fs=1.000, sehingga cutoff desain 0,5 Hz efektif sekitar 1 Hz pada data ini. V2 mengolah ulang raw menggunakan fs=2.000, HPF 0,5 Hz orde 4 dengan SOS untuk stabilitas numerik, lalu coif5 level 5 + SURE soft per level. Tidak ada resampling. Ini kandidat koreksi, bukan bukti kinerja naik. DWT level 5 tidak dinyatakan optimal secara fisiologis.

**Segmen:** panjang 5.000 dan overlap 50% dipertahankan untuk membatasi perubahan. Segmen belum berpusat pada PVC teranotasi. Source paper memakai jendela sekitar R peak PVC; perbedaan masukan/unit evaluasi membuat angka akurasi paper tidak langsung sebanding dengan hasil project.

## Urutan manual

1. `notebooks/06_Audit_Sumber_dan_Preprocessing_V2.ipynb`: prepare → checksum → audit label/sumber → grafik review → preprocessing **development saja**.
2. `notebooks/07_Ablasi_Real_Only_V2.ipynb`: 45 kasus Encoder, 45 PatchTST, 45 Hierarchical; masing-masing sel terpisah.
3. `notebooks/08_Ablasi_Class_Weight_V2.ipynb`: 135 kasus dengan inverse-frequency segmen training fold saja.
4. `notebooks/09_Seleksi_Final_dan_Laporan_V2.ipynb`: seleksi setelah 270 kasus lengkap → 3 model × 5 seed final → test historis → bootstrap/variasi seed/grafik.

Buka kernel Python project. `DEVICE="mps"` untuk Apple Silicon, `"cpu"` bila tidak tersedia. CUDA belum didukung pada ekstensi ini karena checkpoint RNG legacy belum menyimpan state CUDA; jangan menganggap resume lintas-perangkat identik. Jangan jalankan dua notebook training sekaligus pada perangkat yang sama.

Run baru: `results/research_v2/20260930_anatomy_fs2000_v2/`. Semua endpoint juga tersedia melalui `.venv/bin/python -m research_v2 --help`. Contoh satu kasus:

```sh
.venv/bin/python -m research_v2 cv --device mps --arm real_only --model encoder --scenario S01 --fold 1
```

Notebook menyimpan log langsung. Setelah interupsi, jalankan sel yang sama: resume dari akhir epoch terakhir. Kasus lengkap dilewati. Kode/config/input yang berubah ditolak oleh fingerprint; gunakan run baru untuk metode baru. Model final disimpan, bobot CV yang sudah selesai tidak disimpan untuk menghemat disk. Data baru memakai memory map tanpa menyalin seluruh dataset ke RAM sekaligus.

## Desain dan batas interpretasi

- Grid tetap 3 LR × 3 batch, lima fold pasien yang sama; resep optimizer/model/seed dipertahankan pada kedua arm. Perbedaan antar-arm hanya class weighting. LR/batch dipilih melalui mean patient macro-F1; tie-break SD lalu nama arm/ID skenario. Epoch final median lima best epoch.
- Tidak memakai ulang scaler/GAN dari preprocessing lama. Setiap scaler V2 fit training fold; final scaler fit seluruh development. GAN lama hanya pembanding historis, bukan arm setara.
- Seed final 42–46 ditetapkan sebelum evaluasi; seed 42 primer. Semua 15 training final wajib selesai sebelum membaca sinyal test. Lima pengulangan bukan lima fold test, bukan lima pasien independen, bukan pemilihan seed terbaik.
- Test 59 pasien pernah dipakai pada eksperimen lama. Pelaporan tetap menyebut hold-out historis. Perbandingan old-vs-V2 mengubah preprocessing dan augmentasi sekaligus, sehingga tidak boleh menyimpulkan penyebab tunggal.
- CI macro-F1 menggunakan 2.000 bootstrap terstratifikasi pada pasien, terpisah tiap seed. SD seed menggambarkan variasi training. CV yang dipakai memilih konfigurasi bukan estimasi nested CV yang bebas bias seleksi.
- Grafik audit menggunakan satu pasien development per kelas sebagai contoh visual, bukan validasi morfologi klinis seluruh dataset.

## Prioritas setelah V2

1. Review ahli untuk morfologi real/sintetis dan label anatomi; kurasi/annotasi PVC agar target asal IVA tidak dibebankan pada segmen tanpa PVC.
2. Implementasi kandidat GAN dengan delapan kanal independen, QA amplitudo/spektrum/relasi sadapan/keragaman pada training saja. Lalu ablation augmentasi **pada preprocessing V2 yang sama**, protokol baru sebelum melihat test. Tahap ini belum diimplementasikan dan tidak diklaim selesai.
3. Baseline yang sesuai ukuran kohort, ablation arsitektur dan/atau transfer learning yang benar; arsitektur sekarang masih dilatih dari awal.
4. Validasi eksternal/prospektif pada pasien yang belum memengaruhi desain. Tulis keterbatasan jika belum tersedia.
5. Lengkapi naskah sesuai [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8) atau [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) sesuai lingkup studi. Kelayakan Q1 memerlukan kontribusi ilmiah dan validasi; tidak dijamin oleh target F1 tertentu.
