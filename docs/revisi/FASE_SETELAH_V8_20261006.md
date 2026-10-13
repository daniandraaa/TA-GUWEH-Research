# Fase setelah audit V8 — 6 Oktober 2026

**V8 selesai dan valid, tetapi GAN belum memberikan perbaikan yang konsisten terhadap dua kontrol dan kedua recall RVOT.** [Analisis hasil](ANALISIS_HASIL_V8_20261006.md). Tidak menjalankan ulang 41–45 atau membuka test untuk memilih metode berikut.

**Fase B/V9 kini selesai dan diaudit:** notebook 46–50, **10 referensi identik V8 + 20 fit baru** valid. Encoder F1 0,5170 → 0,5311 tetapi recall other turun; Hierarchical F1 0,5081 → 0,4943. Median belum menjadi perbaikan universal. [Analisis V9](ANALISIS_HASIL_V9_20261006.md). Tidak mengulang fase ini; berikutnya fase C, kompatibilitas pretrained/adapter dan protokol fine-tuning terkontrol.

| Urutan | Pertanyaan | Pekerjaan dan batasnya | Status |
|---|---|---|---|
| A | Mengapa baseline V8 berbeda dari V4? | Bandingkan training/scaler/crop dan F1 pada epoch sama; jelaskan pemilihan epoch inner versus outer | Audit selesai: 30 trajectory loss/accuracy cocok, 24 F1 epoch sama identik |
| B | Apakah agregasi pilihan epoch mengurangi sensitivitas terhadap inisialisasi? | Median tiga epoch inner V8 per model/fold; refit real-only dengan recipe sama, outer hanya dinilai sesudah epoch terakhir | Selesai V9: 30/30; kenaikan F1 Encoder disertai penurunan recall other, Hierarchical turun |
| C | Apakah representasi dan fine-tuning membantu real-only? | Audit kandidat ECG-FM/ECG-JEPA, sumber bobot, sampling, lead, durasi dan lisensi; tetapkan from-scratch, linear probe dan fine-tuning pada input setara | Kajian awal sumber primer tersedia; adapter, bobot dan pilot pretrained belum dijalankan |
| D | Apakah augmentasi membantu setelah baseline lebih stabil? | Pisahkan mixed bag/dosis dari kualitas generator, gunakan kontrol real setara dan seed GAN tambahan bila manfaat awal menjanjikan | GAN V7/V8 dibekukan; rasio tidak diperbesar secara otomatis |
| E | Apakah hasil dapat digeneralisasi dan ditulis sebagai kontribusi? | Evaluasi pasien yang lebih independen, ketidakpastian berbasis pasien, hasil positif/negatif, Bab III–V dan batas tanpa ahli ECG | Penulisan hasil yang terverifikasi bisa diteruskan; kohort baru belum tersedia |

Prioritas tetap **real-only dan pemilihan epoch/representasi**, tanpa menghapus pasien, mengganti tiga kelompok anatomi atau memilih seed test. Encoder GAN menunjukkan sinyal positif pada pooled OOF, tetapi recall other turun; Hierarchical GAN turun di mean fold maupun pooled OOF. Keduanya harus ditulis dengan definisi metrik yang sama, tanpa mengganti ukuran hanya untuk mengejar skor.

V8 memakai satu seed GAN per fold dan bank 64 window per kelas; hasilnya berlaku untuk dosis satu penggantian pada bag K4 penuh dan jumlah epoch shared dari real-only inner. Ini belum pengujian semua bentuk GAN atau semua anggaran convergence per arm. Menambah rasio/epoch dapat diuji hanya sebagai hipotesis baru yang dibekukan, bukan solusi yang sudah terbukti.

Protokol, paket/run V9 dan notebook terpisah tersedia. Persiapan/diagnosis serta pengujian CPU sintetis mendahului training; smoke MPS dan training data asli tetap manual. Preserve semua source/runtime/checkpoint V4–V8 agar hasil tidak perlu diulang. Setelah V9, ikuti [rencana audit representasi dan fine-tuning](RISET_REPRESENTASI_DAN_FINE_TUNING_20261006.md), lalu tetapkan protokol sebelum training pretrained baru.
