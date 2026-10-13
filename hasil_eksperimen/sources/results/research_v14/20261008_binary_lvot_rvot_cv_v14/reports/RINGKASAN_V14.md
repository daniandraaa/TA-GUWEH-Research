# Ringkasan V14: model biner khusus vs model 3 kelas digabung (CV development)

Pasangan identik: fold, seed efektif, resep, scaler dan skenario sama dengan V13. Test tidak dibaca. Ambang tetap 0,5. Wilcoxon dua sisi, Holm atas 6 model x arm.

- encoder/baseline (baseline): F1 V13-gabung 0.836 → V14-biner 0.823 (Δ -0.013, Holm p=0.4155); AUC 0.910 → 0.854 (Holm p=1.0000); sensitivitas LVOT 0.753 → 0.702
- encoder/finetune (G02): F1 V13-gabung 0.864 → V14-biner 0.890 (Δ +0.026, Holm p=0.0878); AUC 0.932 → 0.938 (Holm p=1.0000); sensitivitas LVOT 0.798 → 0.825
- patchtst/baseline (baseline): F1 V13-gabung 0.813 → V14-biner 0.748 (Δ -0.066, Holm p=0.5995); AUC 0.920 → 0.858 (Holm p=1.0000); sensitivitas LVOT 0.693 → 0.531
- patchtst/finetune (G02): F1 V13-gabung 0.855 → V14-biner 0.899 (Δ +0.044, Holm p=0.0088); AUC 0.919 → 0.940 (Holm p=0.0873); sensitivitas LVOT 0.754 → 0.830
- hierarchical/baseline (baseline): F1 V13-gabung 0.848 → V14-biner 0.879 (Δ +0.031, Holm p=0.1440); AUC 0.923 → 0.929 (Holm p=1.0000); sensitivitas LVOT 0.760 → 0.772
- hierarchical/finetune (G03): F1 V13-gabung 0.855 → V14-biner 0.881 (Δ +0.026, Holm p=0.0536); AUC 0.925 → 0.929 (Holm p=1.0000); sensitivitas LVOT 0.744 → 0.801

## Keputusan pra-registrasi (arm fine-tuning)

- encoder: **v13_collapsed** (Δ F1 +0.026, Holm p=0.0878)
- patchtst: **v14_binary** (Δ F1 +0.044, Holm p=0.0088)
- hierarchical: **v13_collapsed** (Δ F1 +0.026, Holm p=0.0536)
