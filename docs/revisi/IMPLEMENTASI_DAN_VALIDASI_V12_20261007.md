# Implementasi dan validasi V12 — 7 Oktober 2026

Fase pertama setelah kajian literatur telah dibuat: **identitas beat beranotasi → evaluasi lintas dataset → ekspor kandidat lokal bersyarat**. Notebook **62–68** siap dijalankan pengguna secara manual. Kode berada dalam `research_v12/`, dengan run baru `20261007_beat_identity_v12`; tidak bergantung pada cache encoder V10.

**Belum ada training, pemrosesan waveform atau inference ECG asli V12.** Pengujian memakai array sintetis di direktori temporer. Audit data asli hanya membaca header/anotasi dan menghitung hash. Folder `results/research_v12` belum dibuat saat pemeriksaan ini; tidak ada skor origin baru atau klaim peningkatan ke 0,9.

## Pekerjaan yang selesai

| Komponen | Implementasi |
|---|---|
| Protokol dan identitas eksperimen | Source/runtime/protocol/runner serta dependency source difingerprint; tidak mengubah protokol lama |
| Split sumber | INCART 75 rekaman, 32 grup Holter, lima outer fold dan satu inner grouped split per outer; grup disjoint |
| Label jenis beat | N; V/r; beat lain terdefinisi. Q/non-beat tidak menjadi training normal |
| Fitur | Crop pusat detector 0,8 detik, bentuk/polaritas/energi per lead; pembanding tambahan RR dari tetangga detector |
| Head dan seleksi | Logistic regression L2; scaler/imputer training saja, cap/bobot grup deterministik; C/threshold dipilih pada inner |
| Eksekusi bertahap | Endpoint persiapan, ekstraksi, QA, pilot, CV, analisis, deployment, evaluasi eksternal, ekspor dan status |
| Evaluasi | Semua reference PVC eligible dihitung, termasuk detector miss; unmatched prediksi PVC menjadi FP |
| Audit/visualisasi | Rekonstruksi seluruh metrik evaluasi dari prediksi tersimpan, OOF, CSV/grafik dan bootstrap grup bersyarat |
| Pembekuan sebelum eksternal | CV/split/deployment diverifikasi sebelum MIT-BIH dibuka; tidak memilih ulang threshold dari eksternal |
| Ekspor lokal | Hanya bila gate PASS; kandidat raw 12 lead, mask, fallback tanpa tipe, coverage seluruh 230 development |
| Panduan | [Panduan manual V12](PANDUAN_MANUAL_V12.md), README dan status proyek diperbarui |

Anggaran: **30 inner + 10 outer = 40 fit CV**; **8 pilot** termasuk dalam angka tersebut, bukan tambahan. **Satu fit deployment** dari seluruh INCART sesudah CV. MIT-BIH dan ekspor lokal hanya inference. Ini baseline engineering jenis beat, belum fine-tuning bobot CNN–Transformer atau reproduksi uPVC-Net.

## Bukti pemeriksaan

**24/24 tes lulus**, melalui `.venv/bin/python -m unittest discover -s tests_v12 -v`. Tes mencakup:

- Mapping label, crop tepi, one-to-one matching, denominator recall seluruh PVC, Q dan false positive unmatched.
- Tidak ada irisan grup inner/outer; MIT-BIH 201/202 menjadi grup sama; metadata INCART tanpa ID sumber ditolak.
- Training cap/bobot, pasangan sampel kedua arm, fit imputer/scaler/head tiga kelas pada array sintetis.
- Lead sebenarnya tanpa kanal palsu, preservasi polaritas dan kesetaraan waktu pada sampling berbeda menggunakan sinyal buatan.
- Metrik end-to-end versus conditional, probability invalid, bootstrap, mask/pemilihan kandidat, manifest/run-id.
- Alur **40 fit CV sintetis**, resume tanpa menulis ulang manifest fit lengkap, audit OOF, satu fit deployment sintetis, inference eksternal sintetis dan audit.
- Manifest dengan grup berbeda ditolak meskipun hash internalnya konsisten; fitting baru dibekukan setelah snapshot eksternal; gate FAIL menahan ekspor lokal.
- Eksternal tidak dapat dibuka sebelum preparation/deployment lengkap.

Log: [tes sintetis](../../tmp/v12_synthetic_tests_20261007.log); [hasil mesin](../../tmp/v12_synthetic_test_result_20261007.json). Fontconfig menulis peringatan cache pada lingkungan sandbox; tes/plot sintetis selesai dan exit code 0. Runner manual menggunakan cache Matplotlib project.

**Tujuh notebook lolos validasi JSON, sintaks seluruh sel kode dan setup dari cwd `notebooks/`.** Semua execution count masih kosong dan output belum diisi. Hanya sel setup dijalankan saat verifikasi, tanpa memanggil endpoint: [bukti](../../tmp/v12_notebook_validation_20261007.json).

Audit header/anotasi asli: **75 rekaman, 32 grup**, reference eligible N **150.346**, V/r **20.007**, other **5.475**. Ketiganya tersedia pada semua split. Ini jumlah anotasi setelah aturan crop tepi, bukan jumlah kandidat matched hasil ekstraksi.

| Fold | Grup inner training | Grup inner validation | Grup outer validation |
|---|---:|---:|---:|
| 1 | 19 | 7 | 6 |
| 2 | 19 | 6 | 7 |
| 3 | 20 | 6 | 6 |
| 4 | 19 | 6 | 7 |
| 5 | 20 | 6 | 6 |

Bukti dan hash sumber: [audit metadata](../../tmp/v12_metadata_audit_20261007.json).

**15.070 berkas lama dalam snapshot tetap identik**, total 4.660.334.369 byte; tidak ada file berubah atau hilang. Cakupan meliputi source/protocol lama, runner/notebook yang telah ada, artefak hasil yang tercantum dalam snapshot dan aset INCART. Pemeriksaan ini tidak mengklaim memantau setiap file yang mungkin ada di seluruh project. [Baseline](../../tmp/v12_frozen_before_20261007.json) · [hasil preservasi](../../tmp/v12_preservation_after_20261007.json).

## Batas interpretasi dan langkah pengguna

Baseline publik menguji jenis beat, bukan tiga asal anatomi. Skor head belum dikalibrasi; Q tidak mempunyai label terdefinisi untuk precision. INCART adalah development yang hasil benchmark lamanya telah dibaca. MIT-BIH adalah pengujian transfer detector/head dengan kanal Holter yang berbeda; PASS tidak memvalidasi identitas PVC/sinus lokal secara klinis. Kandidat lokal tetap pseudo-label dan tidak menggunakan kelas anatomi untuk seleksi.

Gate ditetapkan sebagai aturan engineering usulan sebelum hasil dilihat. Bila FAIL, simpan hasil dan analisis kegagalan; jangan mengubah ambang agar ekspor lolos pada run yang sama. Fase ablation input/CNN–Transformer asal anatomi dan evaluasi pasien baru masih menunggu diagnosis ini.

**Mulai [62](../../notebooks/62_V12_Persiapan_Identitas_Beat.ipynb) → 63 → 64**, kernel `.venv`, Shift+Enter. Setelah delapan fit pilot selesai lanjut 65 → 66 → 67; 68 hanya bila 67 PASS. Alur rinci, keluaran dan resume ada pada [panduan manual](PANDUAN_MANUAL_V12.md).
