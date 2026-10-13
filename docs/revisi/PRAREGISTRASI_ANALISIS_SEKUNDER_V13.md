# Pra-registrasi analisis sekunder V13

Ditetapkan **2026-10-07 11:57:39 WIB**, sebelum data test V13 dibuka. Saat dokumen ini dibuat, run `20261008_proposal_baseline_vs_finetune_v13` baru menyelesaikan 5 dari 450 fit CV. Folder `test/` belum ada, seleksi belum dibekukan, dan belum ada model final.

## Endpoint utama (tidak berubah)

Macro-F1 tingkat pasien tiga kelas (LVOT / RVOT cusp / RVOT septal-free-wall-other), sesuai proposal dan `research_v13/protocol.yaml`.

## Analisis sekunder yang ditetapkan

1. **Endpoint biner LVOT vs RVOT.** P(RVOT) = P(RVOT cusp) + P(RVOT other) dari model tiga kelas yang sama. Ambang tetap **0,5**. Metrik: macro-F1, AUC, akurasi, balanced accuracy, sensitivitas LVOT dan RVOT. Dilaporkan untuk CV development (konfigurasi baseline dan fine-tuning terpilih, 5 fold × 3 seed) dan test (tiap seed + ensemble).
2. **Pembanding tebakan mayoritas.** Semua pasien diprediksi RVOT. Akurasinya wajib dilaporkan berdampingan dengan akurasi biner.
3. **Ensemble 5 seed final (42–46).** Rata-rata probabilitas tiap model × skenario. Seluruh lima seed dipakai; tidak ada seleksi seed.
4. **Uji baseline vs fine-tuning pada ensemble.** McNemar exact dan bootstrap pasien terstratifikasi (5.000 sampel, seed 20261008) untuk kedua endpoint, dengan koreksi Holm antar-3-model per endpoint.

Tidak ada ambang, seed, model, atau konfigurasi yang dipilih dari data test.

## Keterbukaan

Gagasan endpoint biner muncul setelah melihat prediksi test historis V4 pada **59 pasien yang sama** (7 Oktober 2026): F1 biner sekitar 0,81–0,89 vs 3 kelas sekitar 0,54–0,61. Karena itu bukti yang tidak terpengaruh test adalah hasil biner pada **CV development V13**. Hal ini wajib disebutkan di Bab III/IV. Endpoint ini perlu persetujuan pembimbing sebagai tambahan proposal.

## Kode yang dikunci (SHA-256)

`analysis_v13/secondary.py` menolak berjalan bila hash di bawah tidak cocok.

- __init__.py: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- secondary.py: f0c8ea5d725c2e1bc65719d97eb0800f288f17c340fb2b9ba90a8d36c871edf6

Menjalankan setelah test V13 lengkap:

```bash
.venv/bin/python -m analysis_v13.secondary
```
