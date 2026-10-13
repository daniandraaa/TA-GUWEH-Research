# Analisis hasil V9 — 6 Oktober 2026

**Notebook 46–50 selesai dan valid. Median epoch memberi sedikit kenaikan macro-F1 Encoder, tetapi mengurangi recall RVOT other pada kedua model dan menurunkan macro-F1 Hierarchical. Aturan ini belum layak dijadikan perbaikan final universal.**

Audit terpisah: [laporan dan grafik](../../results/research_v9/20261006_inner_epoch_median_real_only_v9/analysis_20261006_complete/REPORT.md) · [audit mesin](../../results/research_v9/20261006_inner_epoch_median_real_only_v9/analysis_20261006_complete/audit.json). Paket training, hasil asli dan notebook yang telah dijalankan tidak diubah.

## Pekerjaan yang sudah selesai

| Pekerjaan | Hasil terverifikasi |
|---|---|
| Notebook 46–50 | Semua sel kode dijalankan: 6/6, 5/5, 3/3, 3/3, 4/4; tidak ada error tersimpan pada versi terakhir |
| Persiapan dan kedua smoke MPS | Lulus; smoke memakai satu update disposable per model dan bobotnya dibuang |
| Training baru | 20/20 fit, 308 epoch, 7.130 optimizer update; semua bag real-only |
| Referensi V8 | 10/10; recipe, device, scaler, pasien dan epoch tepat identik; bukan replikasi baru |
| Checkpoint pembanding | 30/30 finite dan sama dengan bobot checkpoint epoch terakhir |
| Data/evaluasi | 230 pasien development, tiga kelas tetap; prediksi seluruh window dipool per pasien |
| Reproduksi analisis notebook 50 | Cases, summary, paired delta, seed summary, pooled OOF dan 2.760 baris prediksi cocok |
| Lintasan training pada epoch bersama V8/V9 | 30/30 loss dan accuracy identik bit demi bit; crop/exposure/anggaran dan scaler valid |
| Integritas hasil lama | Snapshot 1.622 dan 1.903 file terverifikasi tidak berubah |
| Final/historical test | Tidak dibuka pada V9; belum ada model final baru |

Log notebook 49 memuat satu kegagalan awal karena guard disk: tersedia 3,17 GiB, perlu 3,53 GiB. Percobaan berikutnya selesai dan seluruh 20 fit baru lengkap. Tidak perlu mengulang training karena kejadian tersebut. Saat audit tersedia sekitar 5,36 GiB; tidak ada checkpoint lama dihapus oleh audit.

2.760 baris adalah **230 pasien × dua model × dua aturan × tiga seed**, bukan 2.760 pasien independen. Support pasien: LVOT 57, RVOT cusp 103, RVOT septal/freewall/other 70.

## Hasil primer: mean fold/seed macro-F1

| Model | V8 per-seed epoch | V9 median inner | Perubahan F1 | Recall cusp V8 → V9 | Recall other V8 → V9 |
|---|---:|---:|---:|---:|---:|
| Encoder | 0,5170 ± 0,0887 | 0,5311 ± 0,0814 | +0,0141 | 41,37% → 52,56% | 46,19% → 36,67% |
| Hierarchical | 0,5081 ± 0,0958 | 0,4943 ± 0,1068 | −0,0139 | 42,54% → 45,35% | 49,05% → 41,90% |

SD pada tabel adalah variasi deskriptif 15 kasus fold/seed per model, bukan confidence interval atau 15 observasi independen.

Encoder memperbaiki rata-rata F1 dan recall cusp, tetapi kehilangan **9,52 poin persentase recall other**. Hierarchical kehilangan **7,14 poin persentase recall other**, sementara rata-rata F1 turun dan variasi antar-kasus meningkat. Tidak ditemukan kegagalan implementasi yang menjelaskan perubahan tersebut; perubahan checkpoint/epoch memang menghasilkan prediksi yang berbeda.

Pada 70 pasien other yang dinilai masing-masing tiga seed, kesalahan **other → cusp** Encoder bertambah **99 → 121**, Hierarchical **81 → 91**. Angka tersebut menghitung evaluasi berulang dari total 210 evaluasi other, bukan pasien unik. Kenaikan recall cusp tidak boleh dilaporkan seolah seluruh kemampuan lokalisasi membaik.

![Recall kedua RVOT](../../results/research_v9/20261006_inner_epoch_median_real_only_v9/analysis_20261006_complete/rvot_recall_comparison.png)

## Efek antar-seed dan pooled OOF

| Model | Seed | Mean fold-F1 V8 | V9 | Perubahan |
|---|---:|---:|---:|---:|
| Encoder | 42 | 0,5290 | 0,5381 | +0,0091 |
| Encoder | 43 | 0,4872 | 0,5310 | +0,0437 |
| Encoder | 44 | 0,5348 | 0,5242 | −0,0106 |
| Hierarchical | 42 | 0,4936 | 0,4695 | −0,0241 |
| Hierarchical | 43 | 0,5294 | 0,4575 | −0,0718 |
| Hierarchical | 44 | 0,5013 | 0,5557 | +0,0544 |

Encoder naik pada dua seed, Hierarchical hanya satu. Median belum memberikan arah efek positif yang konsisten. Memilih seed 44 Hierarchical saja akan menyembunyikan dua hasil negatif lainnya.

![Perubahan per seed](../../results/research_v9/20261006_inner_epoch_median_real_only_v9/analysis_20261006_complete/paired_seed_macro_f1.png)

| Model | Mean pooled OOF V8 ± SD tiga seed | V9 |
|---|---:|---:|
| Encoder | 0,5270 ± 0,0271 | 0,5480 ± 0,0131 |
| Hierarchical | 0,5412 ± 0,0266 | 0,5177 ± 0,0507 |

Pooled OOF menghitung satu F1 dari seluruh 230 pasien per seed. Mean fold-F1 merata-ratakan lima nilai F1 fold. Karena F1 nonlinear, nilainya dan sebagian arah per seed bisa berbeda; keduanya valid dan sudah direkonstruksi. Ukuran primer tetap mean fold/seed macro-F1 yang telah ditentukan, dengan pooled OOF sebagai pelaporan tambahan.

## Apa yang dapat disimpulkan

V9 menguji **median tiga pilihan epoch inner pada split pasien yang sama**, dengan LR/pooling/batch/scaler/crop/scheduler dan inisialisasi outer yang tetap seperti V8. V9 belum menguji variasi split inner, bentuk representasi baru, pretrained model, seluruh kemungkinan anggaran epoch atau batas maksimum kinerja.

30 kasus median memakai total anggaran 462 epoch, termasuk 154 epoch dari 10 referensi lama; 308 epoch sisanya benar-benar dilatih baru. Kontrol V8 per-seed berjumlah 573 epoch. Karena anggaran berubah per seed sesuai hipotesis, hasil ini mengukur efek **aturan pemilihan epoch**, bukan tambahan GAN atau arsitektur baru.

V8/V9 masih ablation development eksploratif: LR/pooling telah dipilih pada development V4 dan metode berikut diusulkan sesudah membaca beberapa hasil pada development yang sama. Hasilnya tidak dapat dipromosikan menjadi estimasi nested yang sepenuhnya independen. 59 pasien historical test telah pernah dinilai pada versi lama, sehingga tidak dapat dinamai unseen eksternal untuk eksperimen lanjutan.

## Langkah berikutnya

1. **Bekukan V9 sebagai ablation yang selesai.** Tidak mengulang 46–50, memilih seed terbaik, atau menaikkan epoch secara umum. Median belum dipilih sebagai aturan final universal; V8 per-seed tetap kontrol yang terdokumentasi. Belum menentukan satu model final berdasarkan V9 ini saja.
2. **Selesaikan audit kompatibilitas representasi pretrained ECG.** Cocokkan checkpoint/sumber/lisensi, lead, satuan, sampling, durasi dan normalisasi. Native 2000 Hz/2,5 detik tidak boleh langsung dianggap sesuai input model 500 Hz/5–10 detik. Siapkan adapter dan environment terpisah agar runtime lama tetap dapat mereproduksi hasil.
3. **Tetapkan pilot from-scratch versus pretrained.** Bandingkan inisialisasi acak, pretrained encoder dibekukan + head tiga kelas, dan fine-tuning terkontrol pada input yang sama. LR/epoch dipilih dari inner training; outer digunakan untuk penilaian setelah refit. Mulai dari smoke/pilot manual sebelum lima fold penuh.
4. **Perkuat evaluasi dan naskah.** Laporkan recall kedua RVOT, variasi seed, biaya dan keterbatasan label/interpretasi tanpa ahli ECG. Rencanakan evaluasi pasien yang lebih independen serta ketidakpastian berbasis pasien. Tiga kelas dan semua pasien dipertahankan; kasus sulit tidak dihapus untuk meningkatkan skor.

[Kajian dan urutan implementasi representasi/fine-tuning](RISET_REPRESENTASI_DAN_FINE_TUNING_20261006.md). Bobot, adapter dan pilot pretrained belum dijalankan. Audit ini tidak melatih classifier, menjalankan waveform inference atau mengubah hasil asli. Skor saat ini belum membuktikan kinerja maksimum dan belum menjamin 0,9/publikasi Q1.
