# Analisis hasil V2 — 1 Oktober 2026

Audit run `20260930_anatomy_fs2000_v2`: **270/270 CV, 15/15 final, 15/15 test lengkap**. Input frozen lolos checksum; metrik dihitung ulang dari prediksi per pasien. Tidak ada training/inferensi tambahan pada analisis ini.

## Hasil utama

| model | arm | scenario | learning_rate | batch_size | epochs | cv_f1 | cv_sd | test_f1_seed42 | test_accuracy_seed42 | test_auc_seed42 | seed_mean_f1 | seed_sd_f1 | seed_min_f1 | seed_max_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| encoder | weighted_real | S01 | 0.0001 | 16 | 10 | 0.6738 | 0.0274 | 0.6351 | 0.6271 | 0.7816 | 0.6158 | 0.0344 | 0.5773 | 0.6512 |
| patchtst | weighted_real | S05 | 0.0005 | 32 | 15 | 0.6712 | 0.0785 | 0.5664 | 0.5763 | 0.8171 | 0.5876 | 0.0193 | 0.5664 | 0.6137 |
| hierarchical | real_only | S01 | 0.0001 | 16 | 8 | 0.6598 | 0.0643 | 0.5455 | 0.5424 | 0.7430 | 0.5720 | 0.0623 | 0.4991 | 0.6687 |

Encoder tetap model utama dari development CV. Test memakai seed utama 42; mean/SD lima seed dilaporkan terpisah. Seed terbaik pada test tidak menggantikan seed utama. CI 95% seed utama: Encoder 0,514–0,751; PatchTST 0,467–0,671; Hierarchical 0,424–0,662 (bootstrap pasien yang telah dibuat notebook 09). Interval lebar dan beririsan; hasil ini tidak membuktikan perbedaan signifikan antar-model.

## Efek class weighting

| model | arm | scenario | f1_mean | f1_std | epochs |
| --- | --- | --- | --- | --- | --- |
| encoder | real_only | S05 | 0.6655 | 0.0739 | 14.0000 |
| encoder | weighted_real | S01 | 0.6738 | 0.0274 | 10.0000 |
| hierarchical | real_only | S01 | 0.6598 | 0.0643 | 8.0000 |
| hierarchical | weighted_real | S04 | 0.6515 | 0.0670 | 8.0000 |
| patchtst | real_only | S02 | 0.6631 | 0.0410 | 11.0000 |
| patchtst | weighted_real | S05 | 0.6712 | 0.0785 | 15.0000 |

| model | mean_delta | median_delta | wins | losses |
| --- | --- | --- | --- | --- |
| encoder | -0.0031 | 0.0044 | 25 | 20 |
| hierarchical | -0.0020 | 0.0004 | 24 | 21 |
| patchtst | -0.0222 | -0.0179 | 14 | 31 |

Best weighted-vs-best real: Encoder +0,0082, PatchTST +0,0081, Hierarchical −0,0083 macro-F1. Perbandingan ini juga memilih konfigurasi LR/batch berbeda. Analisis pasangan seluruh 45 kasus per model menunjukkan class weighting tidak konsisten membantu; pada PatchTST 31/45 pasangan turun. Kelima fold memiliki training set beririsan, sehingga 45 pasangan bukan 45 replikasi independen. Tidak dihitung p-value naif. Satu seed CV per kasus belum mengukur variasi seed pada proses seleksi.

## Sumber kesalahan

| model | class_name | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| encoder | LVOT | 0.7857 | 0.7097 | 14.0000 |
| encoder | RVOT_cusp | 0.5714 | 0.6000 | 21.0000 |
| encoder | RVOT_septal_freewall_other | 0.5833 | 0.5957 | 24.0000 |
| patchtst | LVOT | 1.0000 | 0.8485 | 14.0000 |
| patchtst | RVOT_cusp | 0.1905 | 0.2581 | 21.0000 |
| patchtst | RVOT_septal_freewall_other | 0.6667 | 0.5926 | 24.0000 |
| hierarchical | LVOT | 0.8571 | 0.7059 | 14.0000 |
| hierarchical | RVOT_cusp | 0.6190 | 0.5417 | 21.0000 |
| hierarchical | RVOT_septal_freewall_other | 0.2917 | 0.3889 | 24.0000 |

PatchTST hanya mendeteksi 4/21 pasien RVOT cusp pada test seed 42; 14/21 berpindah ke kelompok RVOT septal/free-wall/other. Pola ini sudah terlihat pada development OOF: recall RVOT cusp 50/103 = 48,5%. Hierarchical hanya benar 7/24 kelompok RVOT septal/free-wall/other pada test. Encoder lebih seimbang, tetapi OOF recall kelompok itu masih 37/70 = 52,9%.

Development OOF adalah prediksi pasien yang ditahan dari training fold, tetapi konfigurasi dipilih memakai fold yang sama: angka OOF bersifat diagnostik setelah seleksi. Pooled OOF F1 berbeda dari rata-rata F1 fold; gunakan rata-rata fold untuk aturan seleksi yang sudah dibekukan.

Ada **25 pasien development salah pada ketiga model**. Daftar `development_patient_review_priority.csv` menyediakan antrean review; bukan perintah mengganti label. Confidence softmax belum dikalibrasi dan tidak membuktikan diagnosis/label salah. File `development_errors_by_sublocation.csv` perlu dibaca bersama jumlah pasien per sublokasi yang kecil.

## Training dan variasi seed

Encoder mencapai akurasi segmen training sekitar 90,3–92,4%; Hierarchical 88,6–90,3%, sementara akurasi pasien test lebih rendah. Unit evaluasinya berbeda sehingga selisih bukan ukuran overfitting formal, tetapi konsisten dengan kebutuhan memeriksa generalisasi. PatchTST training 62,1–65,8% dan loss akhir 0,676–0,730: kemungkinan resep/representasi/anggaran training belum cocok; belum terbukti bahwa menambah epoch akan memperbaiki test.

Walaupun SD macro-F1 PatchTST paling kecil, prediksi individu berubah pada **46/59 pasien** di antara lima seed (Encoder 31/59, Hierarchical 37/59). Variasi skor rendah tidak sama dengan keputusan pasien yang stabil. Ini hasil diagnostik test historis, bukan alasan memilih seed/ensemble dari test.

Best epoch CV sangat bervariasi: Encoder [10,1,16,2,12]; PatchTST [25,15,12,1,15]; Hierarchical [8,6,20,1,14]. Aturan median digunakan sesuai protokol, tetapi variasi ini perlu diuji melalui CV dengan seed tambahan atau pengaturan jadwal training pada run baru. Jangan mengganti jumlah epoch menurut skor test.

## Audit biner sekunder, eksploratif

Kedua kelompok RVOT dijumlahkan probabilitasnya lalu diprediksi menggunakan threshold tetap 0,5. Untuk PatchTST seed 42, LVOT-vs-RVOT macro-F1 **0,928**, accuracy **56/59 = 94,9%**, AUC **0,986**. Ini mendukung dugaan bahwa kesulitan besar berada pada pemisahan subkelompok RVOT. Analisis dilakukan setelah melihat hasil tiga kelas, bukan endpoint primer/preregistered. Target tiga kelas tetap; angka biner tidak menggantikan macro-F1 tiga kelas dan bukan bukti kelayakan jurnal.

## Perbandingan hasil historis

| model | historical_v1_f1 | v2_f1_seed42 | difference | causal_GAN_effect |
| --- | --- | --- | --- | --- |
| encoder | 0.6367 | 0.6351 | -0.0016 | False |
| patchtst | 0.6775 | 0.5664 | -0.1111 | False |
| hierarchical | 0.5270 | 0.5455 | 0.0185 | False |

Encoder nyaris sama dengan V1, PatchTST turun, Hierarchical naik sedikit. V1 memakai preprocessing lama dan GAN, V2 mengubah preprocessing dan augmentasi sekaligus; tidak boleh menyimpulkan perubahan disebabkan GAN saja atau fs saja. Nama anatomi dikoreksi tanpa perubahan label numerik. Test pernah dievaluasi sebelumnya, sehingga tetap hold-out historis; bukan validasi eksternal.

## Prioritas berikutnya dari bukti ini

1. **Review morfologi PVC dan kurasi segmen pada development.** Periksa pasien yang salah pada tiga model, kualitas sadapan, apakah segmen mengandung PVC, dan konsistensi target anatomi bersama ahli. Jangan otomatis mengubah diagnosis. [Paper sumber](https://pmc.ncbi.nlm.nih.gov/articles/PMC8962834/) memakai jendela sekitar R peak PVC; project memakai potongan 2,5 detik yang belum teranotasi PVC. Detektor R peak saja belum membedakan beat sinus dan PVC.
2. **Uji masukan berpusat pada PVC dan kontribusi pasien yang seimbang.** Tetapkan panjang jendela/aturan agregasi/penanganan segmen pendek sebelum eksperimen, pilih melalui CV development. Sampling saat ini memberi pasien rekaman panjang lebih banyak kontribusi segmen; class weighting belum menyelesaikan pembobotan antar-pasien.
3. **Perbaiki khusus batas dua kelas RVOT.** Bandingkan baseline morfologi/beat dan classifier dua tahap LVOT-vs-RVOT lalu subkelompok RVOT pada run baru; validasi seluruh rangkaian, bukan dua tahap terpisah. Tiga kelas primer dipertahankan. Ambang/kalibrasi harus dipelajari pada validation development, tidak pada test.
4. **Uji resep PatchTST dan stabilitas CV.** Ukur ulang seed pada CV, bobot kelas alternatif/real-only, scheduler, pooling dan kapasitas sesuai anggaran yang ditetapkan. Jangan hanya memilih LR/batch/seed baru yang kebetulan memperbaiki 59 test ini.
5. **GAN sebagai eksperimen tambahan yang lolos QA.** Generator kanal independen, QA morfologi/spektrum/relasi sadapan/keragaman pada training saja. Bandingkan dengan real dan penanganan imbalance pada preprocessing sama. Hasil V2 tidak mendukung klaim GAN wajib dipakai atau GAN lama penyebab tunggal penurunan.
6. **Validasi pasien baru dan kontribusi ilmiah.** F1 tiga kelas masih terbatas, interval lebar, dan test historis. Untuk target publikasi kuat, dokumentasikan kontribusi, dataset/split, ketidakpastian, keterbatasan, serta evaluasi eksternal/prospektif bila tersedia. Status saat ini belum cukup untuk menyatakan model siap dipakai klinis atau dijamin terbit Q1.

Grafik: `confusion_oof_and_test.png`, `class_weighting_matched.png`, `seed_variation.png`. Sumber tabel adalah artefak run yang telah Anda jalankan; kode analisis tersedia pada `analysis_v2/analyze_results.py`.
