# Paket diskusi pembimbing TA GUWEH

## Kesimpulan untuk malam ini
Bukti paling mutakhir yang ditemukan adalah V14 final/test bertanggal sumber 8 Oktober 2026, bukan status lama di PETA_FILE. Semua angka berasal dari artefak asli yang dibaca melalui SSH read-only; tidak ada training/inference atau perubahan pada Mac.

**Endpoint utama V13 tetap lokalisasi tiga kelompok anatomis:** LVOT, RVOT cusp, RVOT septal/free-wall/other. Macro-F1 tiga kelas tidak boleh diganti atau dibandingkan sebagai tugas yang sama dengan F1 biner. Kelompok RVOT other bukan Non-OT.

**CV tiga kelas fine-tuning:** Encoder 0,6531 ± 0,0625; PatchTST 0,6403 ± 0,0590; Hierarchical 0,6418 ± 0,0579. Baseline masing-masing 0,4915; 0,5220; 0,5465. Fine-tuning berarti tuning hiperparameter + GAN, bukan transfer pretrained: protocol pretrained_weights_used=false. Angka baseline PatchTST rinci ada di CSV sumber.

**Biner LVOT vs RVOT (CV):** V13 collapse probabilitas -> V14 direct: Encoder fine-tuning 0,8641 -> 0,8903 (Holm p=0,0878); PatchTST 0,8553 -> 0,8992 (Holm p=0,0088); Hierarchical 0,8546 -> 0,8806 (Holm p=0,0536). Hanya PatchTST memenuhi aturan keputusan V14 yang ditetapkan; ini bukan kemenangan universal arsitektur.

**Test historis PatchTST:** V13 biner 0,8383 ± 0,0181 vs V14 0,8484 ± 0,0406 (5 seed). Ensemble keduanya F1 0,8659; AUC 0,9619; accuracy 0,8983; sensitivitas LVOT 0,8571/RVOT 0,9111. Perbandingan ensemble ΔF1=0, McNemar p=1, CI bootstrap [0,0]. Kenaikan CV tidak membuktikan kenaikan F1 ensemble test.

**Test V13 tiga kelas (mean ± SD 5 seed):** Encoder fine-tuning 0,5722 ± 0,0693; PatchTST 0,5646 ± 0,0582; Hierarchical 0,5614 ± 0,0653. Ensemble tiga kelas berturut-turut 0,6082; 0,5843; 0,5185. Jangan menyebut mean sebagai ensemble atau memilih seed terbaik.

## Isi paket
- sources/: 44 dokumen/protokol/tabel/metrik agregat asli; path, mtime UTC dan SHA-256 di manifest.json.
- comparison_endpoints.csv: 31 baris perbandingan, memisahkan endpoint/tahap/mean-SD/ensemble.
- comparison_subregions.csv: 18 baris metrik per kelompok, dirata-ratakan atas 5 seed, bukan per-pasien.
- SCENARIO_DISKUSI.md: opsi narasi artikel dan pertanyaan keputusan.
- DATASET_DICTIONARY.md dan dataset/: input model nyata, label segmen dan fold anonim (289 subjek; 2.807 window). Bukan dataset mentah 334 rekaman dan bukan mapping identitas.
- RINGKASAN_PEMBIMBING.pdf: ringkasan yang dapat dipilih/dicari teksnya (bukan scan).
- verification.json: hasil pemeriksaan statistik, integritas dataset dan arsip.

## Definisi dan batas interpretasi
Macro-F1 = rata-rata F1 kelas tanpa bobot support. Tiga kelas menggunakan argmax probabilitas pasien; mean pooling window pada V13. Biner collapse: P(RVOT)=P(cusp)+P(other), ambang tetap 0,5. Sensitivitas LVOT/RVOT berarti recall masing-masing kelas; jangan membaca kolom sensitivity umum macro sebagai LVOT. CV mean/SD terdiri dari 15 fold × seed (5 fold, seed 42–44); bukan 15 kohort independen atau CI. Test mean/SD = 5 seed 42–46 pada 59 subjek yang sama. SD dihitung sebagai sample SD; ensemble = mean probabilitas seluruh seed sebelum menghitung metrik, bukan mean skor.

Test 59 subjek sudah pernah dilihat dalam versi historis. Gagasan endpoint biner muncul setelah hasil test V4. Praregistrasi V13/V14 tidak menghapus paparan historis; jangan menyebut untouched/external test. CV development lebih tepat menjadi bukti utama perbandingan biner, tetapi pemilihan grid dan estimasi di CV yang sama bukan nested outer evaluation dan bisa optimistis. Fold/seed berulang tidak independen sepenuhnya; Wilcoxon 15 pasangan bukan pengganti validasi eksternal.

Protokol menyatakan split per pasien, scaler dan GAN dari training fold saja. Paket ini memverifikasi konsistensi metrik/tabel/input, **bukan audit independen menyeluruh** terhadap linkage identitas, overlap segmen, provenance scaler/GAN atau dataset publik. Klaim bebas leakage/patient-independent harus menunggu audit tersebut. Input window berlabel origin bukan beat PVC yang sudah divalidasi ahli. Pembanding kriteria klinis otomatis belum divalidasi ahli; CI selisih AUC V14 dengan aturan klinis melintasi nol, bukan bukti superioritas klinis.

## Privasi dan izin
Pengguna menegaskan izin berbagi dataset deidentified dengan pembimbing. Identitas hospital_id pada metadata sumber diganti ANON0001 dst secara konsisten pada label/fold; mapping tidak diekspor. Diagnosis.xlsx, file raw ECG dengan nama sumber, patient_predictions dan source cohort tidak dimasukkan. Waveform klinis tetap sensitif dan mungkin memiliki risiko reidentifikasi walaupun tanpa nama: akses hanya pembimbing/tujuan riset yang disetujui. Lisensi dataset asli/consent/ethics belum diverifikasi mandiri; izin pengguna tidak otomatis membuktikan hak redistribusi publik. Tidak ada unggah Drive pada tugas ini.

## Tabel perbandingan
| Versi | Endpoint | Tahap | Model | Arm | F1 mean ± SD | Ensemble |
| --- | --- | --- | --- | --- | --- | --- |
| V13 | 3class localization | development CV | encoder | finetune | 0.6531 ± 0.0625 | - |
| V13 | 3class localization | development CV | encoder | baseline | 0.4915 ± 0.0525 | - |
| V13 | 3class localization | development CV | hierarchical | finetune | 0.6418 ± 0.0579 | - |
| V13 | 3class localization | development CV | hierarchical | baseline | 0.5465 ± 0.0543 | - |
| V13 | 3class localization | development CV | patchtst | finetune | 0.6403 ± 0.0590 | - |
| V13 | 3class localization | development CV | patchtst | baseline | 0.5220 ± 0.0751 | - |
| V13 | 3class localization | historical holdout | encoder | baseline | 0.4264 ± 0.0892 | 0.4205 |
| V13 | binary collapsed3class | historical holdout | encoder | baseline | 0.7671 ± 0.1244 | 0.9063 |
| V13 | 3class localization | historical holdout | encoder | finetune | 0.5722 ± 0.0693 | 0.6082 |
| V13 | binary collapsed3class | historical holdout | encoder | finetune | 0.8349 ± 0.0281 | 0.8659 |
| V13 | 3class localization | historical holdout | hierarchical | baseline | 0.3923 ± 0.1351 | 0.5210 |
| V13 | binary collapsed3class | historical holdout | hierarchical | baseline | 0.7218 ± 0.1721 | 0.8224 |
| V13 | 3class localization | historical holdout | hierarchical | finetune | 0.5614 ± 0.0653 | 0.5185 |
| V13 | binary collapsed3class | historical holdout | hierarchical | finetune | 0.8318 ± 0.0384 | 0.8659 |
| V13 | 3class localization | historical holdout | patchtst | baseline | 0.2918 ± 0.1521 | 0.4669 |
| V13 | binary collapsed3class | historical holdout | patchtst | baseline | 0.6073 ± 0.2394 | 0.8110 |
| V13 | 3class localization | historical holdout | patchtst | finetune | 0.5646 ± 0.0582 | 0.5843 |
| V13 | binary collapsed3class | historical holdout | patchtst | finetune | 0.8383 ± 0.0181 | 0.8659 |
| V13 | binary collapsed3class | development CV | encoder | baseline | 0.8359 ± 0.0621 | - |
| V14 | binary direct | development CV | encoder | baseline | 0.8231 ± 0.1692 | - |
| V13 | binary collapsed3class | development CV | encoder | finetune | 0.8641 ± 0.0407 | - |
| V14 | binary direct | development CV | encoder | finetune | 0.8903 ± 0.0436 | - |
| V13 | binary collapsed3class | development CV | patchtst | baseline | 0.8134 ± 0.0806 | - |
| V14 | binary direct | development CV | patchtst | baseline | 0.7479 ± 0.2024 | - |
| V13 | binary collapsed3class | development CV | patchtst | finetune | 0.8553 ± 0.0300 | - |
| V14 | binary direct | development CV | patchtst | finetune | 0.8992 ± 0.0280 | - |
| V13 | binary collapsed3class | development CV | hierarchical | baseline | 0.8477 ± 0.0580 | - |
| V14 | binary direct | development CV | hierarchical | baseline | 0.8788 ± 0.0433 | - |
| V13 | binary collapsed3class | development CV | hierarchical | finetune | 0.8546 ± 0.0437 | - |
| V14 | binary direct | development CV | hierarchical | finetune | 0.8806 ± 0.0360 | - |
| V14 | binary direct | historical holdout | patchtst | finetune | 0.8484 ± 0.0406 | 0.8659 |
