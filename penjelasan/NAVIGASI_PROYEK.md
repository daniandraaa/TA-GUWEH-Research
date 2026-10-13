# Navigasi proyek

| Lokasi | Fungsi / cara baca |
|---|---|
| `01_EDA_Preprocessing.py`, `02_Baseline_CNN_Transformer.py`, `03_Finetune_CNN_Transformer.py` | Skrip awal/historis; bukan hasil terbaru |
| `src/` | Artifact/signature, data, model/training/evaluasi, GAN dan pelaporan bersama |
| `research_v2/`, `research_v3/`, `research_v4/` | Asal canonical cache, kandidat model dan patient-bag/MIL; dibutuhkan oleh versi lanjut |
| `research_v5/`–`research_v12/` | Iterasi, audit/ablation/eksplorasi sebelumnya, bukan endpoint final artikel otomatis |
| `research_v13/protocol.yaml`, `common.py`, `data.py`, `experiment.py`, `gan_final.py`, `analysis.py`, `explain.py` | Desain 3class proposal, baseline vs GAN + grid, seleksi CV, final/test/statistik/SHAP |
| `research_v14/protocol.yaml`, `experiment.py` | Direct-binary development CV dan keputusan berpasangan terhadap V13 collapse |
| `research_v14_final/run.py` | Tahap final/test PatchTST yang dipilih CV; pilihan dibekukan dan hash diperiksa |
| `analysis_v13/secondary.py`, `analysis_v14/` | Analisis sekunder collapse/ensemble, calibration/DCA dan kriteria klinis |
| `results/research_v13/20261008_proposal_baseline_vs_finetune_v13/` | Artefak 3class dan `reports_secondary/`; teks asli terpilih |
| `results/research_v14/20261008_binary_lvot_rvot_cv_v14/` | Perbandingan CV dan `reports/decision.json` |
| `results/research_v14/20261008_binary_patchtst_final_test_v14/` | Final/test historis dan perbandingan ensemble |
| `docs/revisi/` | Praregistrasi/desain/manual; status lama mungkin mendahului hasil terbaru |
| `tests*/`, `notebooks/`, `scripts/`, `configs/` | Uji, notebook tanpa output, runner dan konfigurasi; tidak dijalankan untuk publikasi ini |
| `hasil_eksperimen/` | Ringkasan pembimbing, CSV agregat, PDF dan 44 sumber audit diskusi |
| `PACKAGING_*` | Audit ZIP sebelum penambahan panduan repositori |

Path historis raw/dataset/checkpoint di README asli tidak berarti file tersebut termasuk paket. Lihat manifest, bukan asumsi dari dokumentasi lama.
