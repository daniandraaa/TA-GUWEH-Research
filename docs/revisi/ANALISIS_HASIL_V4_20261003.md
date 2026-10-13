# Analisis hasil V4 — 3 Oktober 2026

**Notebook 20–27 selesai tanpa error tersimpan.** Run `20261002_patient_mil_hybrids_v4` telah diaudit dari prediksi pasien, konfigurasi, riwayat epoch, provenance scaler, pooling seluruh segment, dan checkpoint. Fingerprint V4 serta patch MPS valid; hasil sebelum patch tetap utuh. Tidak ada training baru atau inference sinyal ECG oleh audit ini.

## Pekerjaan yang selesai

| Tahap | Hasil |
|---|---|
| Preparation/fold/scaler | 230 pasien development, 59 historical test; tidak ada irisan ID |
| Diagnosis native | 45/45 fit tiga model, lima fold, tiga seed |
| Pilot | 50/50 kasus; menjadi bagian full grid |
| Full grid | 450/450 kasus: lima model × dua pooling × sembilan LR/batch × lima fold |
| Stabilitas konfigurasi terpilih | 75/75 hasil; 25 memakai ulang CV seed 42 dan 50 fit tambahan |
| Final | 25/25 checkpoint: lima model × lima seed, dilatih pada seluruh development |
| Historical test | 25/25 evaluasi pada 59 pasien yang sama, tanpa k-fold test |
| Visualisasi | Notebook 27 dan laporan/grafik hasil lengkap |

Total training unik protokol adalah 570 fit, termasuk diagnosis, grid, tambahan stability dan final. Audit menghitung ulang enam metrik untuk 450 CV, 45 diagnosis, 50 stability baru dan 25 test; **selisih maksimum dengan metrik tersimpan = 0**. Seluruh 25 checkpoint final berhingga dan dapat dimuat dengan key/shape yang sesuai. Hyperparameter dan median epoch final direkonstruksi dari CV, cocok dengan seleksi tersimpan. Evaluasi memakai seluruh segment tiap pasien; bobot pooling berjumlah satu dan mean pooling seragam. Training bag mengunjungi setiap pasien training tepat sekali per epoch. Pemeriksaan numerik ini tidak memvalidasi morfologi/diagnosis klinis sinyal.

Audit dapat diulang tanpa training:

```bash
.venv/bin/python -m analysis_v4.audit_completed
```

Kode audit berada di luar paket training dan patch MPS yang sudah dibekukan. Output terpisah di `results/research_v4/20261002_patient_mil_hybrids_v4/analysis_20261003_complete/`; `final_audit.json` memuat ringkasan dan `verified_input_sha256.json` mencatat artefak yang diperiksa. Audit memuat bobot di CPU untuk memeriksa checkpoint, tetapi tidak menjalankan ulang prediksi sinyal.

## Hasil lima model

Semua angka berikut **macro-F1 pasien tiga kelas**, bukan training accuracy. CV adalah mean ± SD lima fold pada konfigurasi terpilih; test lima seed adalah mean ± SD training seed pada 59 pasien yang sama. Seed primer tetap 42, bukan seed dengan test tertinggi.

| Model | CV terpilih | Test seed 42 | Test mean ± SD lima seed |
|---|---:|---:|---:|
| Encoder | 0,6596 ± 0,0855 | 0,5709 | 0,5942 ± 0,0343 |
| PatchTST | 0,6448 ± 0,0708 | 0,6311 | 0,6088 ± 0,0809 |
| **Hierarchical, model primer** | **0,6616 ± 0,0797** | **0,6198** | **0,5421 ± 0,0845** |
| Residual multiscale | 0,6269 ± 0,0766 | 0,5814 | 0,5068 ± 0,0534 |
| Lead–Time | 0,5996 ± 0,0544 | 0,5728 | 0,4992 ± 0,0452 |

Konfigurasi primer: Hierarchical, **mean pooling, S05, LR 5e-4, segment budget 32 / patient minibatch 8, 13 epoch final, seed 42**. Akurasi test 0,6102; AUC macro OVR 0,7940. Selisih CV Hierarchical–Encoder hanya 0,0020; ini pilihan menurut aturan seleksi yang sudah ditetapkan, belum bukti keunggulan yang kuat. Macro-F1 pada gabungan 230 prediksi OOF Hierarchical adalah 0,6738; angka itu berbeda dari rata-rata F1 lima fold 0,6616 karena F1 bukan metrik linear.

Target 0,9 belum tercapai. Dua arsitektur tambahan belum memberikan peningkatan pada evaluasi ini. Hasil sekarang tidak membuktikan batas maksimum dataset atau keluarga CNN–Transformer. Yang sudah diuji adalah protokol, representasi, kapasitas dan grid V4 yang tercatat.

## Temuan yang paling menentukan langkah berikutnya

**Pemisahan dua kelompok RVOT masih menjadi kesulitan utama.** Pada Hierarchical seed 42, LVOT benar 13/14 (recall 92,9%), RVOT cusp benar 9/21 (42,9%), dan RVOT septal/freewall/other benar 14/24 (58,3%). Dari 23 kesalahan, 16 merupakan pertukaran dua kelompok RVOT: sepuluh cusp diprediksi sebagai kelompok RVOT lainnya dan enam sebaliknya. Jadi memperbaiki akurasi LVOT saja tidak menyelesaikan endpoint tiga kelas.

**Stabilitas final belum memadai.** Prediksi Hierarchical berubah antar-lima-seed pada 44/59 pasien (74,6%); Encoder 32/59, PatchTST 37/59, Residual 47/59 dan Lead–Time 51/59. Angka ini berarti ada setidaknya satu seed dengan prediksi berbeda, bukan persentase pasien yang pasti salah. Pada tiga konfigurasi lama, seluruh konfigurasi terpilih tidak kehilangan kelas pada fold/seed stability, tetapi variasi final tetap besar. Configurasi lain dalam grid masih mempunyai collapse; tidak dihapus dari laporan.

**Perubahan V3→V4 belum konsisten antar-model.** Mean lima-seed Encoder meningkat 0,5580→0,5942 dan PatchTST 0,5747→0,6088. Hierarchical meningkat pada seed 42, 0,5837→0,6198, tetapi mean lima-seed turun 0,5783→0,5421. Jangan menyimpulkan V4 selalu lebih baik dari satu seed. V4 sekaligus mengubah loss/aggregasi pasien, sampling bag dan recipe training, sehingga perbandingan ini belum mengisolasi satu penyebab.

**Ada kelompok development yang konsisten sulit.** Pada selected OOF seed 42, 12/230 pasien salah pada kelima model dan 37/230 salah pada sedikitnya empat model. Semua 12 kasus yang salah pada lima model berasal dari kedua kelompok RVOT. Dari 37 kasus, 21 berada pada RVOT cusp, 15 pada kelompok RVOT lainnya dan satu LVOT. Daftar untuk pemeriksaan teknis ada di `development_oof_error_triage.csv`, dengan sublokasi dan prediksi tiap model. Ringkasan sublokasi ada di `development_sublocation_error_summary.csv`; AnteriorSeptal memiliki 8/20 kasus salah pada sedikitnya empat model dan LC 11/57. Ini daftar pemeriksaan, bukan alasan otomatis membuang pasien atau mengganti label.

**Ketidakpastian pasien masih lebar.** Bootstrap terstratifikasi 5.000 kali untuk checkpoint primer memberi interval persentil 95% macro-F1 sekitar **0,5006–0,7336**. Perbandingan berpasangan masing-masing model dengan Hierarchical pada seed 42 semuanya mempunyai interval selisih yang melintasi nol. Analisis ini post-hoc pada test historis dengan bobot model tetap; tidak mencakup ketidakpastian training seed, perubahan protokol atau populasi eksternal. CV terpilih memakai fold yang sama untuk tuning dan estimasi, sehingga belum nested CV. SD fold/seed bukan interval kepercayaan pasien.

## Prioritas sesudah V4

1. **Audit representasi pada development terlebih dahulu.** Gunakan 12/37 pasien sulit di atas untuk memeriksa metadata sublokasi, integritas 12 sadapan, amplitude, windowing dan jumlah/bentuk kompleks yang masuk ke model. Periksa juga pasien yang mudah pada kelas/sublokasi sama. Label sumber tetap dipertahankan; tanpa anotasi tambahan, attention atau deteksi QRS otomatis tidak dinyatakan sebagai anotasi PVC yang tervalidasi. Hasil pemeriksaan harus menentukan apakah masalah utama berasal dari representasi window, informasi label, atau keduanya.
2. **Buat ablation baru yang mengisolasi perubahan.** Pada manifest pasien yang sama, bandingkan satu perubahan tiap tahap: representasi window versus kompleks yang disejajarkan secara teknis, mean versus attention, normalisasi, serta jadwal/epoch training yang dipilih hanya dari development. Pertahankan baseline V3/V4 dan beberapa seed. Uji kecil di development sebelum full grid berikutnya. Ensemble/pretraining dapat menjadi kandidat yang ditetapkan dan dipilih lewat validation development; jangan memilih komposisi/seed memakai skor 59 pasien historical test.
3. **Lengkapi kontribusi proposal dan evaluasinya.** GAN terkoreksi belum dilatih dalam V4; lakukan pemeriksaan mutu sintetis serta perbandingan real-only/GAN dengan recipe yang sebanding bila komponen itu tetap dalam tujuan TA. Perkuat evaluasi dengan outer/nested patient CV atau kohort origin-label eksternal yang benar-benar tersedia. Benchmark deteksi beat INCART tidak menggantikan validasi classifier tiga kelompok. Tuntaskan tabel, analisis per kelas, ablation, batasan dan naskah bersama pembimbing.

Mulai dari pemeriksaan development dan desain ablation baru, bukan mengulang notebook 20–26 yang sudah selesai atau mengubah run V4. V4 menjadi pembanding lengkap yang dipertahankan. Data test historis sudah pernah diamati sejak versi sebelumnya; hasilnya dilaporkan sebagai eksplorasi historis. Belum ada bukti yang menjamin skor 0,9 atau penerimaan Q1.
