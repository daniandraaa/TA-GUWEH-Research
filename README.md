# TA GUWEH — paket riset untuk diskusi pembimbing

**Repositori PRIVATE, bukan untuk redistribusi publik.** Salinan kode proyek sebenarnya yang telah disanitasi, hasil agregat V13/V14, dan panduan diskusi. Tidak dilakukan training/inference ulang dan tidak mengubah proyek di Mac. Data waveform tetap sensitif walau identitas diganti anonim; tidak ada lisensi MIT atau izin redistribusi publik yang ditambahkan.

## Mulai di sini (pembimbing)
1. Baca [ringkasan PDF](hasil_eksperimen/RINGKASAN_PEMBIMBING.pdf) dan [hasil serta batas interpretasi](hasil_eksperimen/README.md).
2. Buka [31 perbandingan endpoint](hasil_eksperimen/comparison_endpoints.csv) dan [18 hasil subregion](hasil_eksperimen/comparison_subregions.csv); jangan menyamakan 3class, binary, CV, holdout, mean seed, dan ensemble.
3. Ikuti [navigasi proyek](penjelasan/NAVIGASI_PROYEK.md), [arsitektur](penjelasan/ARSITEKTUR_ENDPOINT.md), [interpretasi](penjelasan/INTERPRETASI_HASIL.md), dan [skenario artikel](penjelasan/SKENARIO_ARTIKEL.md).
4. Untuk audit asal-usul dan menjalankan ulang: [provenance & batas](penjelasan/PROVENANCE_BATAS.md), [reproduksibilitas](penjelasan/REPRODUKSIBILITAS.md), [README packaging](PACKAGING_README.md).

## Hasil yang perlu dibedakan
Endpoint utama V13: LVOT, RVOT cusp, RVOT septal/free-wall/other. Other **bukan Non-OT**. Fine-tuning V13 adalah GAN + grid hiperparameter, bukan pretrained transfer. CV 3class F1 fine-tuning: Encoder 0,6531; PatchTST 0,6403; Hierarchical 0,6418. F1 test historis lebih rendah; lihat tabel lengkap, bukan hanya angka terbaik.

Endpoint sekunder biner: LVOT vs gabungan RVOT. Hanya PatchTST fine-tuning memenuhi aturan perbandingan CV V14 (0,8553 -> 0,8992; Holm p≈0,0088). Pada ensemble test historis V13/V14 F1 sama 0,8659. Ini tidak membuktikan superioritas universal atau generalisasi klinis.

## Kode dan data
Struktur proyek berada langsung di root: `src/`, `research_v2/`–`research_v14/`, `research_v14_final/`, `analysis_v*/`, `configs/`, `scripts/`, `tests*/`, `docs/`, `notebooks/`, dan hasil teks terpilih di `results/`. [README asli](README_PROJECT_ASLI.md) dipertahankan; tautan absolut Mac/status historis di dalamnya tidak dianggap petunjuk portabel atau status terbaru.

Git **tidak** menyimpan `.npy`, checkpoint/weights, ECG mentah, prediksi/ukuran individual, secrets atau environment. Dataset input nyata dan tabel individual anonim ada hanya di ZIP release privat [supervisor-discussion-2026-10](https://github.com/daniandraaa/TA-GUWEH-Research/releases/tag/supervisor-discussion-2026-10), `TA_GUWEH_Project_Sharing.zip` (634949991 bytes). SHA-256: `dc6f5e70a51174cb2c597f3c5cb6ba797b77d21ec81a95761c722291766cf19a`.

ZIP berisi 3354 file sumber packaging, bukan mirror lengkap: development 230 pasien/2274 window, historical test 59 pasien/533 window, 12×5000 float32. Checkpoint, GAN cache, raw ECG dan identitas asli tidak disertakan. Metadata anonim mengubah frozen signature lama; **clone atau unzip tidak menjamin full run siap**. `PACKAGING_MANIFEST.json` dan `PACKAGING_SHA256SUMS.txt` menjelaskan ZIP asal, bukan checksum tree Git yang sudah ditambah dokumen/diubah README.

## Akses dan izin
Pengguna menegaskan izin berbagi data deidentified dengan pembimbing. Consent/ethics/lisensi belum diverifikasi secara independen. Pembimbing memerlukan undangan akun GitHub yang dikonfirmasi; belum ada collaborator ditambahkan. Jangan mengubah visibility ke public atau memindahkan dataset ke repository publik terkait.
