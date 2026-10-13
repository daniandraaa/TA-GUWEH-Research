# Ringkasan hasil V13 (baseline vs fine-tuning)

Angka utama: macro-F1 tingkat pasien. CV = 5 fold x 3 seed pada 230 pasien development; test = 59 pasien hold-out historis, 5 seed final.

## Test 20% (mean ± SD 5 seed)

- encoder / baseline: F1 0.4264 ± 0.0892; akurasi 0.4983; sensitivitas 0.5417; spesifisitas 0.7485; AUC 0.7254
- encoder / finetune: F1 0.5722 ± 0.0693; akurasi 0.5627; sensitivitas 0.6040; spesifisitas 0.7740; AUC 0.7708
- hierarchical / baseline: F1 0.3923 ± 0.1351; akurasi 0.4847; sensitivitas 0.5024; spesifisitas 0.7324; AUC 0.7290
- hierarchical / finetune: F1 0.5614 ± 0.0653; akurasi 0.5593; sensitivitas 0.5988; spesifisitas 0.7726; AUC 0.7534
- patchtst / baseline: F1 0.2918 ± 0.1521; akurasi 0.4475; sensitivitas 0.4389; spesifisitas 0.7103; AUC 0.6819
- patchtst / finetune: F1 0.5646 ± 0.0582; akurasi 0.5797; sensitivitas 0.6115; spesifisitas 0.7810; AUC 0.7728

## Uji signifikansi

- CV encoder, fine-tuning terpilih (GAN + grid): Δ=+0.1616, Wilcoxon p=0.0001, Holm p=0.0002 → signifikan (α=0.05)
- CV encoder, efek GAN saja (LR/batch sama dengan baseline): Δ=+0.0382, Wilcoxon p=0.0946, Holm p=0.1892 → tidak signifikan (α=0.05)
- CV patchtst, fine-tuning terpilih (GAN + grid): Δ=+0.1183, Wilcoxon p=0.0004, Holm p=0.0009 → signifikan (α=0.05)
- CV patchtst, efek GAN saja (LR/batch sama dengan baseline): Δ=-0.0837, Wilcoxon p=0.0413, Holm p=0.1238 → tidak signifikan (α=0.05)
- CV hierarchical, fine-tuning terpilih (GAN + grid): Δ=+0.0953, Wilcoxon p=0.0026, Holm p=0.0026 → signifikan (α=0.05)
- CV hierarchical, efek GAN saja (LR/batch sama dengan baseline): Δ=+0.0307, Wilcoxon p=0.3028, Holm p=0.3028 → tidak signifikan (α=0.05)
- Test encoder seed 42: ΔF1=+0.1341 [CI95 -0.0023, +0.2679], McNemar p=0.8388, Holm p=0.8388
- Test patchtst seed 42: ΔF1=+0.4328 [CI95 +0.3119, +0.5488], McNemar p=0.0730, Holm p=0.2189
- Test hierarchical seed 42: ΔF1=+0.3225 [CI95 +0.2107, +0.4325], McNemar p=0.2005, Holm p=0.4010
- Friedman antar-3-model (baseline, CV): χ²=5.200, p=0.0743
- Friedman antar-3-model (finetune, CV): χ²=2.533, p=0.2818

Catatan: pasangan fold x seed tidak sepenuhnya independen (fold berbagi data training); test 59 pasien pernah diamati pada eksperimen lama sehingga disebut hold-out historis.
