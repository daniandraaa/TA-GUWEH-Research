# Arsitektur dan endpoint

Alur V13: cache window 12 lead × 5000 sample pada 2000 Hz → scaler training fold → sampling patient bag (4 window per pasien saat training) → CNN temporal → representasi Transformer → probabilitas window → mean pooling probabilitas pasien → evaluasi patient macro-F1. Inference memakai chunk segmen sesuai protokol; unit hasil tetap pasien, bukan setiap window dianggap subjek baru.

Tiga backbone kode V3/V4 yang dipakai V13: **encoder** (CNN + Transformer encoder), **patchtst** (patch dari fitur CNN + proyeksi + Transformer, pooling dengan bobot patch), **hierarchical** (Transformer lokal lalu downsampling dan Transformer tingkat kedua). Lihat `research_v3/models.py` dan `research_v4/models.py`; nama PatchTST tidak otomatis berarti reproduksi identik model paper. Eksperimen lain di V4 bukan otomatis arm V13.

Baseline V13: real-only LR 0,001, batch budget 32 segmen. Fine-tuning: GAN per training fold, satu dari empat window diganti sintetis dengan kelas sama, grid LR [0,0001; 0,0005; 0,001] × batch [16;32;64]. Protokol menyatakan validation tidak diaugmentasi dan pretrained_weights_used=false. AdamW, maksimum 100 epoch, patience 15. Fold 1–5, seed CV 42–44, seed final 42–46. Seleksi mean patient macro-F1 CV. GAN + grid berubah bersama, sehingga efek keseluruhan tidak dapat diatribusikan ke GAN saja.

**3class utama V13:** label 0 LVOT; 1 RVOT cusp; 2 RVOT septal/free-wall/other. Kelas 2 bukan Non-OT. Argmax tiga probabilitas pasien menentukan kelas.

**Biner sekunder V13:** P(RVOT)=P(kelas1)+P(kelas2); P(LVOT)=P(kelas0), threshold 0,5. Ini bukan model dilatih biner.

**Biner V14:** model dilatih dua kelas, map {0:0,1:1,2:1}; pasien/fold/seed/arm dipasangkan dengan V13. Aturan pilihan per-model fine-tuning: delta F1 CV positif dan Wilcoxon Holm p<0,05; selain itu tetap V13 collapse. Hanya PatchTST memenuhi aturan. V14 final/test hanya PatchTST, bukan semua arsitektur di-test ulang.
