# Analisis hasil V5 — 4 Oktober 2026

**Notebook 29–32 selesai tanpa error tersimpan. Seluruh 30 training baru selesai: Hierarchical 15/15 dan Encoder 15/15. Kedua smoke MPS lulus. Keputusan dari development: pertahankan K=4; K=8 tidak memenuhi kriteria prioritas yang ditetapkan sebelum training.**

Audit tambahan menghitung ulang enam metrik dan recall kelas pada 60 hasil (30 kandidat dan 30 kontrol), memeriksa seluruh window validation, epoch terpilih, identitas pasien/fold/seed, serta tabel perbandingan dan pooled OOF. Seluruh hasil cocok. Sebanyak 300 hash artefak pembanding dan 3.098 artefak hasil V4 tetap valid. Notebook yang telah dieksekusi serta laporan asli tidak ditulis ulang; audit tambahan tidak melatih model atau membaca sinyal test.

## Hasil utama development

Angka berikut adalah **rata-rata macro-F1 lima fold × tiga seed**, bukan skor test atau accuracy training. Kontrol memakai recipe selected V4 pada seed 42, 43, 44; K=8 mempertahankan batch delapan pasien dan meningkatkan kapasitas maksimal segmen per update dari 32 menjadi 64.

| Model | K=4 | K=8 | Perubahan F1 | Seed dengan mean fold naik | Median rasio waktu fit K8/K4 |
|---|---:|---:|---:|---:|---:|
| Hierarchical, mean pooling | 0,6525 | 0,6367 | −0,0158 | 0/3 | 1,51× |
| Encoder, attention pooling | 0,6534 | 0,6427 | −0,0107 | 1/3 | 1,54× |

Hierarchical naik pada empat dari 15 pasangan fold/seed, turun pada sebelas. Encoder naik pada sembilan pasangan, tetapi penurunan pada enam pasangan lain lebih besar sehingga rata-ratanya tetap turun. Jumlah pasangan yang naik tidak menggantikan ukuran utama mean macro-F1. Fold/seed saling terkait dan tidak diperlakukan sebagai 15 pengamatan independen; hasil ini tidak diklaim signifikan secara statistik.

| Model | Seed | Mean fold K4 | Mean fold K8 | Selisih |
|---|---:|---:|---:|---:|
| Hierarchical | 42 | 0,6616 | 0,6505 | −0,0112 |
| Hierarchical | 43 | 0,6462 | 0,6319 | −0,0143 |
| Hierarchical | 44 | 0,6498 | 0,6278 | −0,0220 |
| Encoder | 42 | 0,6596 | 0,6449 | −0,0147 |
| Encoder | 43 | 0,6516 | 0,6577 | +0,0061 |
| Encoder | 44 | 0,6491 | 0,6256 | −0,0235 |

Mean pooled OOF macro-F1 tiga seed juga turun: Hierarchical **0,6597 → 0,6401**, Encoder **0,6574 → 0,6471**. Pooled OOF setiap seed mencakup 230 pasien development unik. Mean fold dan pooled OOF dihitung dengan cara berbeda sehingga tidak harus sama.

## Kelas yang terdampak

Recall berikut dihitung dari pooled OOF setiap seed, lalu dirata-ratakan pada tiga seed. Ini bukan tiga kohort pasien baru.

| Model | Kelompok | K4 | K8 | Perubahan poin persentase |
|---|---|---:|---:|---:|
| Hierarchical | LVOT | 80,12% | 77,78% | −2,34 |
| Hierarchical | RVOT cusp | 58,90% | 65,37% | +6,47 |
| Hierarchical | RVOT septal/freewall/other | 58,57% | 48,10% | **−10,48** |
| Encoder | LVOT | 81,87% | 83,04% | +1,17 |
| Encoder | RVOT cusp | 61,17% | 62,78% | +1,62 |
| Encoder | RVOT septal/freewall/other | 54,76% | 49,52% | **−5,24** |

K=8 meningkatkan recall cusp tetapi merugikan kelompok RVOT lainnya. Rata-rata recall kedua RVOT turun 2,00 poin untuk Hierarchical dan 1,81 poin untuk Encoder. Perubahan ini menunjukkan trade-off antar-kelas, belum membuktikan label salah, kegagalan detector, atau penyebab fisiologis tertentu. Tidak ada pasien yang dikeluarkan dan nama/anggota kelas tetap.

Tidak ada kelas prediksi yang hilang pada hasil checkpoint terpilih dari 60 kasus. Beberapa epoch awal bisa mengalami collapse sementara; nol collapse pada hasil terpilih tidak berarti setiap epoch selalu baik.

## Training dan biaya

| Model | Arm | Median best epoch | Median epoch selesai | Median waktu per epoch |
|---|---|---:|---:|---:|
| Hierarchical | K4 | 13 | 28 | 3,70 s |
| Hierarchical | K8 | 10 | 25 | 5,64 s |
| Encoder | K4 | 14 | 29 | 4,43 s |
| Encoder | K8 | 16 | 31 | 6,92 s |

Semua pasien training dilihat tepat satu kali per epoch, komposisi kelas sesuai training fold, dan jumlah update sesuai batch delapan pasien. Perbedaan hasil tidak berasal dari batch pasien yang tanpa sengaja menjadi empat. Total elapsed fit kandidat yang dicatat sekitar 47,3 menit untuk Hierarchical dan 56,1 menit untuk Encoder; ini akumulasi waktu fit tersimpan, bukan estimasi waktu seluruh sesi notebook. Rasio waktu dipengaruhi jumlah window serta epoch yang dijalankan dan kondisi perangkat, bukan benchmark perangkat yang terkontrol sepenuhnya.

Contoh Encoder K8 fold1 seed42: accuracy training bag mencapai **0,9185** pada epoch35, sementara accuracy validation seluruh window **0,5652**. Checkpoint dipilih pada epoch20 dengan validation macro-F1 **0,6548**. Angka training sekitar 0,9 karena itu tidak menggantikan performa validation. Training menggunakan subset window dan mode train, validation seluruh window dan mode eval; perbedaan tersebut membuat selisih ini indikator untuk pemeriksaan generalisasi, bukan pengukuran tunggal penyebab overfitting.

## Keputusan dan pekerjaan berikut

**K4 tetap menjadi kontrol untuk fase berikutnya.** K8 tidak memenuhi kriteria karena mean F1 turun pada kedua model, Hierarchical tidak naik pada satu pun mean seed, Encoder hanya naik pada satu seed, dan rata-rata recall RVOT turun. Tidak perlu mengulang notebook 29–31 atau melatih model final K8 untuk menguji test historis.

Eksperimen ini menguji jumlah window pada recipe tertentu. Ia belum membuktikan bahwa K4 optimal untuk semua model/recipe, bahwa nilai 0,65 adalah batas maksimal, atau bahwa target 0,9 dapat dicapai. K4 juga hanya mengambil maksimal empat window per kunjungan pasien; window dapat berubah antar-epoch. Hasil negatif ini tetap menjadi ablation yang berguna untuk Bab IV.

Langkah berikut mengikuti fase C yang telah disusun: **lengkapi GAN yang dikoreksi sesuai objektif proposal, diawali pilot mutu pada training fold sebelum classifier real+GAN**. Periksa keragaman, memorisasi, distribusi amplitudo/spektrum dan hubungan sadapan, lalu tetapkan kontrol jumlah update/exposure serta loss pasien sebelum perbandingan. Recipe K4 menjadi baseline; hasil GAN lama dengan preprocessing/label historis tidak dicampur. GAN belum terbukti memperbaiki skor dan tidak diasumsikan mampu mencapai 0,9. Implementasi notebook baru untuk fase ini belum dibuat.

Regularisasi/normalisasi atau pretraining CNN–Transformer masih hipotesis lanjutan untuk generalisasi/pemisahan kedua RVOT. V5 tidak mengisolasi pengaruh normalisasi atau kemampuan representasi, sehingga perubahan tersebut tidak disebut perbaikan yang sudah terbukti. Bila diuji, gunakan run baru dan kontrol satu faktor, tanpa memilih dari test historis. Pretraining diperlukan bila istilah fine-tuning dimaksudkan sebagai adaptasi bobot pretrained; V4/V5 saat ini training dari awal dengan tuning hiperparameter.

Model primer V4 dan hasil historical test tetap: Hierarchical seed42 macro-F1 **0,6198**, mean lima seed **0,5421 ± 0,0845**. V5 tidak melatih final atau menghasilkan skor test baru. Revisi Bab III–V dapat berjalan sekarang; penguatan evaluasi masih memerlukan pemisahan inner/outer selection yang benar dan/atau kohort baru yang kompatibel. Mengganti split tidak menjadikan pasien test yang sudah diamati sebagai validasi eksternal.

## Status yang sudah dituntaskan

- [x] Notebook 29: persiapan dan kedua smoke MPS nyata.
- [x] Notebook 30: 15 fit Hierarchical.
- [x] Notebook 31: 15 fit Encoder.
- [x] Notebook 32: perbandingan 30 kandidat versus 30 kontrol serta grafik PNG/PDF.
- [x] Audit metrik, kelas, exposure pasien, pooling seluruh window, OOF, hash dan preservasi hasil.
- [x] Keputusan development: pertahankan K4, K8 belum diprioritaskan.
- [ ] Pilot GAN terkoreksi, ablation classifier sesuai proposal, evaluasi diperkuat dan Bab III–V.

Audit dapat diulang tanpa training dan tanpa menimpa laporan asli:

```bash
.venv/bin/python -m analysis_v5.audit_completed
```

Artefak audit: `results/research_v5/20261003_window_bag_ablation_v5/analysis_20261004_complete/`: `audit.json`, `training_diagnostics.csv`, `per_class_pooled_oof_by_seed.csv`, `per_class_pooled_oof_mean.csv`, `development_patient_transitions.csv` dan manifest preservasi laporan/notebook. Perubahan pasien bersifat deskriptif; pasien yang muncul pada seed berbeda bukan pengamatan independen dan daftar ini tidak digunakan untuk eksklusi.

[Grafik asli K4/K8](../../results/research_v5/20261003_window_bag_ablation_v5/reports/k4_k8_development.png) · [panduan V5 sebagai referensi](PANDUAN_MANUAL_V5.md) · [fase penyelesaian](FASE_SETELAH_V4_20261003.md).
