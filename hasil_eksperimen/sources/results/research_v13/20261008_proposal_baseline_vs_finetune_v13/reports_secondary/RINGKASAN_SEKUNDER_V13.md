# Analisis sekunder V13 (dipra-registrasi)

Endpoint utama tetap macro-F1 tiga kelas (lihat reports/RINGKASAN_V13.md). Bagian ini sekunder.

Pembanding tebakan mayoritas (semua RVOT): akurasi 0.763, macro-F1 0.433, AUC 0,5.

## Test: rata-rata 5 seed dan ensemble

- encoder/baseline: 3 kelas F1 0.426 (ensemble 0.420); biner F1 0.767 (ensemble 0.906), AUC biner 0.910, sensitivitas LVOT 0.757
- encoder/finetune: 3 kelas F1 0.572 (ensemble 0.608); biner F1 0.835 (ensemble 0.866), AUC biner 0.949, sensitivitas LVOT 0.843
- hierarchical/baseline: 3 kelas F1 0.392 (ensemble 0.521); biner F1 0.722 (ensemble 0.822), AUC biner 0.901, sensitivitas LVOT 0.586
- hierarchical/finetune: 3 kelas F1 0.561 (ensemble 0.518); biner F1 0.832 (ensemble 0.866), AUC biner 0.947, sensitivitas LVOT 0.843
- patchtst/baseline: 3 kelas F1 0.292 (ensemble 0.467); biner F1 0.607 (ensemble 0.811), AUC biner 0.836, sensitivitas LVOT 0.357
- patchtst/finetune: 3 kelas F1 0.565 (ensemble 0.584); biner F1 0.838 (ensemble 0.866), AUC biner 0.950, sensitivitas LVOT 0.843

## CV development (endpoint biner, 5 fold × 3 seed)

- encoder/baseline (baseline): F1 0.836 ± 0.062; AUC 0.910; sensitivitas LVOT 0.753
- encoder/finetune (G02): F1 0.864 ± 0.041; AUC 0.932; sensitivitas LVOT 0.798
- hierarchical/baseline (baseline): F1 0.848 ± 0.058; AUC 0.923; sensitivitas LVOT 0.760
- hierarchical/finetune (G03): F1 0.855 ± 0.044; AUC 0.925; sensitivitas LVOT 0.744
- patchtst/baseline (baseline): F1 0.813 ± 0.081; AUC 0.920; sensitivitas LVOT 0.693
- patchtst/finetune (G02): F1 0.855 ± 0.030; AUC 0.919; sensitivitas LVOT 0.754

## Baseline vs fine-tuning (ensemble)

- encoder, 3 kelas: ΔF1 +0.188 [CI95 +0.070, +0.297], McNemar p=0.3593, Holm p=1.0000
- encoder, biner LVOT vs RVOT: ΔF1 -0.040 [CI95 -0.100, +0.000], McNemar p=0.5000, Holm p=1.0000
- patchtst, 3 kelas: ΔF1 +0.117 [CI95 +0.006, +0.226], McNemar p=0.7744, Holm p=1.0000
- patchtst, biner LVOT vs RVOT: ΔF1 +0.055 [CI95 -0.084, +0.213], McNemar p=1.0000, Holm p=1.0000
- hierarchical, 3 kelas: ΔF1 -0.003 [CI95 -0.125, +0.119], McNemar p=0.6476, Holm p=1.0000
- hierarchical, biner LVOT vs RVOT: ΔF1 +0.044 [CI95 -0.053, +0.162], McNemar p=1.0000, Holm p=1.0000

Catatan: gagasan endpoint biner muncul setelah melihat prediksi test V4 pada 59 pasien yang sama; bukti yang tidak terpengaruh test adalah hasil CV development di atas.
