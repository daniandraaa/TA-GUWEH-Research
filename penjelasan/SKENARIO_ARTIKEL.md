# Skenario penulisan artikel

## A — Utama mengikuti proposal (paling konservatif)
Patient-level lokalisasi tiga kelas V13, tiga backbone baseline vs GAN+grid; binary V13/V14 menjadi analisis sekunder. Nyatakan fine-tuning bukan transfer pretrained. Laporkan CV dan historical test terpisah, semua model/arm dan subgroup, bukan cherry-pick. Efek GAN dan grid tidak terpisah; laporan GAN saja G08 tidak signifikan setelah Holm. Keputusan pembimbing: nomenklatur anatomi, istilah tuning, batas klaim.

## B — Artikel/addendum biner
Direct-binary vs collapse berpasangan, CV development sebagai dasar keputusan per-model. Sorot PatchTST namun ungkap model lain tidak lolos dan ensemble test tidak meningkat. Tidak memilih ulang model dari test. Butuh holdout/kohort baru, threshold/epoch locked dan audit independen untuk clinical generalization. Jelaskan gagasan biner setelah paparan V4 dan sejarah preregistrasi.

## C — Metodologi dan robustness
Fokus granularitas label, heterogenitas RVOT other, seed variability, sampling 2000 Hz, patient pooling, GAN+grid confounding dan reuse holdout. Lanjutkan hanya setelah persetujuan: expert beat/morfologi validation, nested evaluation/cohort eksternal, ablation real-only grid vs GAN dengan anggaran setara, audit pipeline dan aturan klinis. Jangan retuning epoch 5 V14 setelah melihat holdout.

Lihat [skenario diskusi asli](../hasil_eksperimen/SCENARIO_DISKUSI.md) untuk pertanyaan keputusan dan sumber statistik. Struktur artikel: latar/pertanyaan → unit pasien/label/split/arms → tabel CV utama → test historis deskriptif → binary sekunder/subgroup/ensemble → limitations/ethics → validasi lanjut. Tidak menjanjikan publikasi Q1 atau universal winner.
