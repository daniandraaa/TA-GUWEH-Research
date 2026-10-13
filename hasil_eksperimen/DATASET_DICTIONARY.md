# Dataset model nyata dan provenance

Izin: pengguna mengonfirmasi dataset deidentified boleh dibagikan kepada pembimbing. Salinan ini bukan izin redistribusi publik. Tidak ada nama, MRN/hospital_id, tanggal klinis, berkas Diagnosis.xlsx, mapping identitas atau prediksi individu di paket.

## Berkas canonical yang dipakai V13/V14
Sumber waveform: `/Users/daniandraaa/Desktop/TA GUWEH/results/research_v2/20260930_anatomy_fs2000_v2/data/{development,test}/X.npy`.
V13/V14 mewarisi cache ini melalui V3/V4; hash development X.npy identik dengan protocol V13/V14.

| Split | Pasien | Window | Shape | Tipe | Bytes X.npy |
|---|---:|---:|---|---|---:|
| development | 230 | 2274 | (2274,12,5000) | little-endian float32 | 545760128 |
| test historis | 59 | 533 | (533,12,5000) | little-endian float32 | 127920128 |

Urutan baris segments.csv sama persis dengan axis 0 X.npy; tidak mengurutkan ulang sinyal. Waveform denoised native 2000 Hz, 2,5 detik, 12 lead; satuan amplitudo perlu verifikasi sumber sebelum analisis klinis. Urutan lead dokumentasi project: I, II, III, aVR, aVL, aVF, V1–V6. Preprocessing dokumentasi: high-pass 0,5 Hz, coif5 level5 SURE soft; scaler fold **tidak** diterapkan permanen pada ekspor ini.

## Dictionary metadata yang sudah disanitasi
- anonymous_id: ANON0001 dst; ID lokal anonim konsisten kedua split dan fold. ID hospital sumber dihapus. Mapping hanya di memori saat sanitasi, tidak disimpan.
- label: 0=LVOT; 1=RVOT_cusp; 2=RVOT_septal_freewall_other. Biner: 0 -> LVOT, 1/2 -> RVOT.
- start_sample: offset awal window pada rekaman sumber (sampel, bukan tanggal/jam).
- valid_samples: panjang nyata window sebelum padding; pada cache saat inspeksi 5000 seluruh window.
- folds.json: fold, train_anonymous_ids, validation_anonymous_ids; fold 1–5 asli, bukan split baru.

Cohort 289: development LVOT57/cusp103/other70; test LVOT14/cusp21/other24. Total LVOT71/cusp124/other94. Sumber lokal lebih besar memiliki 334 berkas per folder sinyal; 5 Summit dan 40 unknown dikecualikan sesuai dokumen, bukan data hilang. Label historis Non-OT dikoreksi; jangan dipakai lagi.

## Inventaris sumber yang tidak diekspor
- Diagnosis.xlsx: 1 file, 17353 bytes; metadata diagnosis dan potensi linkage identitas, tidak dibuka/diekspor.
- PVCVTRawECGData: 334 CSV, 571691199 bytes.
- PVCVTECGData: 334 CSV, 1476858228 bytes.
- preprocessed_data: 357 file, 2135770462 bytes (336 CSV, 2 NPY, 18 PNG, 1 tanpa ekstensi).
Inventaris berbasis filesystem read-only; jumlah file tidak otomatis jumlah pasien unik.

## Asal data dan hak
Dokumen project PANDUAN_REVISI_V2 merujuk descriptor Zheng et al., Scientific Data 2020 https://www.nature.com/articles/s41597-020-0440-8 dan https://pmc.ncbi.nlm.nih.gov/articles/PMC7090065/ . Jumlah 334/12-lead/2000 Hz cocok, tetapi checksum archive asal belum diverifikasi. Studi Zheng 2022 kohort 545 tidak boleh disamakan dengan lokal 334. Lisensi, consent dan ethics sharing belum ditemukan terverifikasi pada dokumen yang ditinjau. Konfirmasi pemilik data/lisensi tetap diperlukan sebelum ekspor publik.

Sinyal ECG dapat mengandung ciri biologis reidentifikasi walau tidak memiliki header identitas. Distribusi terbatas kepada pembimbing yang diberi izin; tanpa tautan publik. Tidak mengekspor seluruh raw dataset; berkas canonical model input + label/fold anonim cukup untuk pengujian input dan grouping, tetapi bukan reproduksi penuh training karena GAN/checkpoint/model execution environment tidak disertakan.
