# Pra-registrasi V14 final dan test: PatchTST biner

Ditetapkan **2026-10-08 15:59:58 WIB**, setelah keputusan CV V14 dan **sebelum** model final dilatih atau test dibaca.

## Dasar

Aturan pada [pra-registrasi V14](PRAREGISTRASI_V14_BINER_20261008.md) memilih **PatchTST biner** (CV F1 0,855 → 0,899; Δ +0,044; Holm p = 0,0088). Encoder dan Hierarchical tetap memakai V13 (3 kelas digabung).

## Model final

- PatchTST, fine-tuning G02 (LR 1e-4, batch 32), augmentasi GAN final V13 (lolos QC) dengan dua kelompok RVOT digabung.
- Dilatih pada 230 pasien development; scaler di-fit hanya pada development.
- **Epoch = 5**, yaitu median best epoch 15 fit CV (aturan sama dengan V13). Catatan jujur: sebaran best epoch bimodal (2–5 dan 11–23); aturan tetap dipakai tanpa diubah.
- Seed 42–46, tanpa seleksi seed.

## Evaluasi test (59 pasien hold-out historis)

- Per seed dan ensemble 5 seed, ambang tetap 0,5: macro-F1, AUC, akurasi, sensitivitas LVOT/RVOT, kalibrasi.
- Perbandingan dengan V13 PatchTST 3 kelas digabung (ensemble): McNemar exact dan bootstrap selisih F1 (5.000 kali, seed 20261008).
- Selisih AUC terhadap tiga kriteria klinis (bootstrap) dan decision curve (LVOT positif).
- **Hasil test tidak dipakai untuk mengubah model, epoch, ambang, atau memilih seed.** Angka utama paper tetap CV development; test dilaporkan sebagai hold-out historis.

## Kode yang dikunci (SHA-256)

- __init__.py: 715005fc098c82f0745d691774dd5db6451a1dcb6eb40068479ef36b4906db2c
- run.py: 28a0cf534ddb64ea2d8b608a5649b1f1dd13225b2a72bd694cc8235a79ae5ff8
