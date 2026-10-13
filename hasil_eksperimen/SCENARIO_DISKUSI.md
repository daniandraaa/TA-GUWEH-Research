# Skenario diskusi artikel dengan pembimbing

## A — Pertahankan artikel utama sesuai proposal (rekomendasi paling konservatif)
Pertanyaan: apakah tuning hiperparameter + GAN memperbaiki tiga model CNN–Transformer untuk tiga kelompok origin? Endpoint utama tetap patient macro-F1 tiga kelas V13; biner V13/V14 sekunder. Gunakan semua baseline dan arm terpilih, bukan satu model terbaik saja.

Temuan: delta fine-tuning CV signifikan per laporan Wilcoxon/Holm; Friedman antar-model tidak signifikan (baseline p=0,0743; fine-tuning p=0,2818). GAN saja pada G08 tidak signifikan setelah Holm. Karena fine-tuning mengubah GAN + grid, jangan mengatribusikan seluruh peningkatan kepada GAN. Test historis lebih rendah daripada CV dan bervariasi antar seed; paparkan subgroup RVOT secara eksplisit.

Keputusan yang diminta: setujui endpoint/nama anatomi terkoreksi; apakah istilah fine-tuning diganti tuning hiperparameter; apakah analisis biner masuk sebagai sekunder dalam satu artikel?

## B — Fokus klinis biner sebagai artikel terpisah/addendum
Pertanyaan: direct-binary vs collapse model tiga kelas, pada pasien/fold/resep berpasangan. Sorot PatchTST G02: CV F1 0,8553 -> 0,8992, delta 0,04395, Holm p=0,00881. Encoder dan Hierarchical tidak lolos aturan; tetap collapse. Keputusan ini berbasis per-model, bukan ranking universal arsitektur.

Trade-off: tugas lebih mudah dan RVOT subregion dihilangkan. Test 59 historis hanya mendukung deskriptif: ensemble V13/V14 identik F1 0,8659. Encoder baseline V13 ensemble biner 0,9063 adalah angka deskriptif test, **bukan alasan memilih ulang pemenang dari test**. Rata-rata seed berbeda dengan ensemble. Majority RVOT accuracy 0,7627, macro-F1 0,4327: jangan menjual accuracy tanpa pembanding ini.

Keputusan: bolehkah biner menjadi kontribusi utama melalui addendum eksplisit? Butuh kohort eksternal baru atau holdout benar-benar baru, locked threshold/epoch, dan audit independen sebelum klaim clinical generalization. Praregistrasi setelah paparan V4 harus diungkapkan.

## C — Paper metodologi/robustness, bukan mengejar angka tertinggi
Narasi: granularitas label, variabilitas seed, perubahan endpoint, sampling 2000 Hz, patient pooling, hyperparameter/augmentation confounding, dan reuse historical holdout. Tabel utama menunjukkan CV dan test terpisah, per-kelas dan ensemble terpisah. Pertanyaan: mengapa LVOT relatif mudah tetapi cusp/other sulit? Apakah label other heterogen dan window memuat PVC/sinus campur?

Langkah lanjutan untuk disetujui (tidak dijalankan di paket ini): validasi ahli beat/morfologi, audit grouping/pipeline/fold scaler/GAN, nested evaluation atau cohort eksternal, ablation real-only grid vs GAN dengan anggaran sama, validasi pengukur aturan klinis. Jangan mengubah epoch 5 V14 sesudah melihat holdout; sebaran epoch bimodal merupakan keterbatasan, bukan izin retuning test.

## Agenda keputusan singkat
1. Tetapkan unit dan endpoint artikel (patient, 3class utama vs binary sekunder).
2. Sepakati transparansi test historis dan paparan gagasan biner.
3. Tentukan tabel wajib: endpoint CSV, subgroup CSV, majority, mean/SD dan ensemble.
4. Sepakati audit dan validasi eksternal yang belum ada, bukan menjanjikan universal winner/Q1.
5. Pastikan izin dataset, lisensi sumber, akses Drive terbatas dan tanpa mapping identitas.

## Referensi internal yang dapat diaudit
Semua path di bawah relatif terhadap `/Users/daniandraaa/Desktop/TA GUWEH/` dan salinan ada di sources/:
- docs/revisi/PANDUAN_MANUAL_V13.md (desain; kalimat status persiapan lama bukan status hasil kini).
- docs/revisi/PRAREGISTRASI_ANALISIS_SEKUNDER_V13.md (paparan V4 dan definisi collapse).
- docs/revisi/PRAREGISTRASI_V14_BINER_20261008.md (aturan CV).
- docs/revisi/PRAREGISTRASI_V14_FINAL_TEST_20261008.md (epoch, seed, holdout historis).
- results/research_v13/20261008_proposal_baseline_vs_finetune_v13/reports/cv_summary.csv dan bab4_tabel_test.csv.
- results/research_v13/20261008_proposal_baseline_vs_finetune_v13/reports_secondary/bab4_tabel_sekunder.csv.
- results/research_v14/20261008_binary_lvot_rvot_cv_v14/reports/cv_comparison.csv.
- results/research_v14/20261008_binary_patchtst_final_test_v14/reports/test_v13_vs_v14_patchtst.csv dan paired_test_v13_vs_v14.json.
