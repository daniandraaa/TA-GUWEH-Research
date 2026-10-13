# Pra-registrasi V14: model biner khusus LVOT vs RVOT

Ditetapkan **2026-10-08 13:24:00 WIB**, sebelum training V14 dimulai.

## Pertanyaan

Apakah model yang **dilatih langsung untuk 2 kelas** (LVOT vs RVOT) lebih baik daripada **model 3 kelas V13 yang probabilitas dua kelas RVOT-nya digabung**?

## Desain

- **Data:** 230 pasien development V13 dengan fold 1–5 yang sama. **Test 59 pasien tidak dibaca** (`test_pool_read: false`).
- **Label:** LVOT = 0; RVOT cusp dan RVOT septal/free-wall/other = 1.
- **Skenario per model:** baseline (LR 1e-3, batch 32, tanpa GAN) dan fine-tuning yang dipilih V13 dari CV 3 kelas (Encoder G02, PatchTST G02, Hierarchical G03). Tidak ada Grid Search baru.
- **Pasangan:** seed CV 42–44 dengan seed efektif `seed + 100 × fold`, resep, scaler dan GAN fold V7 yang identik dengan V13. Setiap fit V14 berpasangan dengan fit V13 pada pasien validasi yang sama.
- **Augmentasi GAN:** 1 dari 4 window diganti window sintetis dari sisi yang sama. Untuk pasien RVOT, window sintetis cusp dan other digabung.
- **Jumlah:** 3 model × 2 arm × 5 fold × 3 seed = **90 fit**.

## Analisis dan aturan keputusan

- Metrik utama: macro-F1 biner tingkat pasien, ambang 0,5. Juga dilaporkan AUC, balanced accuracy dan sensitivitas LVOT/RVOT.
- Uji: Wilcoxon dua sisi berpasangan (15 pasangan fold × seed), koreksi Holm atas 6 model × arm.
- **Aturan (arm fine-tuning, per model):** pakai V14 bila rata-rata selisih F1 > 0 **dan** p Holm < 0,05. Selain itu tetap pakai V13 yang digabung, karena lebih sederhana dan model finalnya sudah ada.
- Model final dan evaluasi test/eksternal untuk V14 hanya dibuat bila aturan di atas memilih V14, dengan pra-registrasi terpisah.

## Keterbukaan

Ide endpoint biner muncul setelah melihat prediksi test V4 (lihat [pra-registrasi analisis sekunder V13](PRAREGISTRASI_ANALISIS_SEKUNDER_V13.md)). V14 sengaja hanya memakai CV development agar perbandingan ini tidak dipengaruhi test.

## Kode yang dikunci (SHA-256)

- __init__.py: e52ab3ea29a5ce72fc5ad1d9a0c19072c676fbb4b4a34d7e7a14837186225707
- __main__.py: d5bf01e815b508d0edbe3e48ec751258dd3713f4df0130196c1cae1990cd330f
- common.py: f9794418868244b9a42917d9381e1019d04958409b8c6bb3433f2e655df79bbd
- experiment.py: 0c4dd77d20127cee4bbcb6407dc27e3b65a1ac81a95c286dfb3f08f50c37a42d
- protocol.yaml: 963d16ae807a09ddd74eaf6b1272da0ba198cb8df05f1136af0215747efa8a9f

Hash juga tercatat otomatis di `results/research_v14/20261008_binary_lvot_rvot_cv_v14/protocol.json`; run menolak lanjut bila kode berubah.
