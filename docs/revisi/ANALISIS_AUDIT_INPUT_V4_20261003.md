# Analisis notebook 28 — audit input development selesai

**Perkembangan 4 Oktober 2026:** ablation V5/notebook 29–32 telah selesai, 30 fit baru dan 30 kontrol diaudit. K8 tidak memenuhi kriteria sehingga baseline K4 tetap. [Analisis hasil V5](ANALISIS_HASIL_V5_20261004.md). Dokumen ini memuat dasar/desain audit input dari 3 Oktober.

**Diperiksa 3 Oktober 2026. Notebook 28 selesai pada 230/230 pasien development dan menghasilkan 12 pratinjau, tanpa error tersimpan.** Pemeriksaan tambahan mencocokkan hash raw dengan sumber preprocessing V2, seluruh kolom tabel kualitas dengan JSON per pasien, serta hitungan kandidat setiap window dengan pusat kandidat tersimpan. Fingerprint V4/MPS kompatibel valid dan 3.098 artefak yang dicatat pada audit hasil V4 tetap sama. Tidak ada training classifier baru dalam audit ini.

## Temuan yang terverifikasi

| Pemeriksaan | Hasil | Makna |
|---|---:|---|
| Pasien development / window native | 230 / 2.274 | Seluruh kohort development diperiksa |
| Durasi window / sampling | 2,5 detik / 2.000 Hz | Representasi V4 tetap |
| Pasien dengan kanal numerik konstan | 0 | Tidak ditemukan RMS ≤ 1e-8 pada raw/cache |
| Pasien dengan dua kanal identik persis | 0 | Tidak ditemukan duplikasi numerik persis |
| Overlap cache yang tidak konsisten / gap | 0 / 0 | Rekonstruksi prefix cache lulus |
| Window tanpa kandidat konsensus lengkap | 6 dari 2.274 (0,264%) | Terjadi pada tiga pasien, bukan bukti ketiadaan PVC |
| Pasien dengan flag teknis detector | 16 | Dua dari 37 pasien sulit; flag tidak menentukan eksklusi |
| Median agreement dua detector | 0,9726 | Kesepakatan detector, bukan ketepatan klinis |
| Median jumlah window per pasien | 8; rentang 1–46 | Jumlah konteks tersedia berbeda antar-pasien |
| Pasien mempunyai lebih dari 4 / 8 window | 193 / 103 | Mendukung pengujian ukuran bag training |
| Median durasi raw / tail tidak tertutup cache | 11,296 / 0,587 detik | Tail belum dipakai; belum terbukti penyebab kesalahan |

Tiga pasien dengan window tanpa kandidat: **ANON0103** (RVOT cusp, 1/22 window), **ANON0124** (RVOT cusp, 3/7), dan **ANON0209** (RVOT septal/freewall/other, 2/24). Ketiganya salah pada empat dari lima model selected OOF, tetapi hanya mencakup tiga dari 37 pasien sulit. Pemeriksaan ini belum mendukung penjelasan bahwa mayoritas kesalahan berasal dari kegagalan deteksi kandidat.

Daftar review berisi 37 pasangan hard/control dengan 32 kontrol unik: 36 pasangan cocok kelas dan sublokasi, satu cocok kelas saja, lima memakai ulang kontrol. Median agreement hard 0,9825 versus kontrol berpasangan 0,9706; median selisih per pasangan adalah 0. Angka ini deskriptif, kelompok ditetapkan setelah kesalahan OOF diamati, dan kontrol yang dipakai ulang bukan pengamatan independen.

## Implikasi untuk model

Audit tidak menemukan masalah numerik dasar yang cukup untuk menjelaskan seluruh kesalahan RVOT. Hasil ini belum menyatakan kualitas ECG klinis sempurna, label salah, ataupun performa sudah maksimal. Sumber label tetap dipertahankan; tidak ada pasien yang dihapus atau diganti labelnya.

Kandidat konsensus bukan anotasi PVC. Sinyal dapat berisi sinus atau artefak meskipun kandidat ditemukan. Sebanyak 12 gambar hanya pratinjau teknis, belum review ahli. Kolom raw pada gambar memakai lima detik pertama sedangkan cache memakai window tengah; keduanya bisa berbeda waktu. Statistik raw memakai seluruh rekaman sementara statistik cache memakai prefix yang tertutup window. Rasio amplitudo/derivative keduanya tidak dipakai sebagai bukti kausal efek denoising atau normalisasi. Unit file belum diasumsikan mV.

**Prioritas berikut: uji maksimal 4 versus 8 window per pasien saat training.** V4 mengambil paling banyak empat window acak tanpa pengembalian pada setiap kunjungan pasien per epoch; evaluasi memakai seluruh window pasien. Pada median pasien delapan window, K=4 memakai separuh window per kunjungan. K=8 memakai seluruh delapan window pada pasien median tersebut. Sampling acak pada epoch berikutnya dapat memperluas cakupan; empat window tidak berarti empat yang sama dipakai sepanjang training. Window overlap juga tidak menjadi informasi independen.

Pada pasien dengan 46 window, ekspektasi teoritis proporsi window yang pernah terambil selama 13 epoch dengan sampling uniform independen adalah sekitar 69,4% untuk K=4 dan 91,7% untuk K=8, menurut `1 − (1 − min(K,n)/n)^epoch`. Ini ilustrasi untuk horizon final Hierarchical V4, **bukan log penggunaan window aktual** atau bukti bahwa penambahan K memperbaiki skor. Epoch hasil CV tiap fold dapat berbeda.

## Desain ablation pertama

| Komponen | Kontrol V4 | Kandidat baru |
|---|---|---|
| Model / pooling | Hierarchical mean; Encoder attention | Sama |
| Maksimal window training per pasien | 4 | 8 |
| Batch pasien per update | 8 | **Tetap 8** |
| Kapasitas maksimal segmen per update | 32 | 64, sebagai konsekuensi K bertambah |
| LR | Hierarchical 5e-4; Encoder 1e-4 | Sama |
| Fold / seed | Lima fold tetap; 42, 43, 44 | Sama |
| Preprocessing, scaler, loss, optimizer, scheduler | Recipe selected V4 | Sama |
| Validation | Seluruh window validation; seleksi development | Sama |

Menahan kapasitas segmen pada 32 ketika K bertambah menjadi 8 akan menurunkan batch pasien menjadi empat dan mencampur efek ukuran bag dengan batch pasien. Karena itu kapasitas harus naik menjadi 64. Biaya komputasi per update bertambah; lakukan smoke/pilot manual dan catat waktu serta penggunaan memori. Jika perangkat tidak mampu, dokumentasikan dan rancang ulang sebelum full run; jangan diam-diam mengubah batch pasien.

Anggaran: **2 model × 5 fold × 3 seed = 30 fit baru** untuk satu arm K=8. Sebanyak 30 hasil kontrol memakai selected stability V4 setelah recipe dan tautan/hash provenance diverifikasi. Total perbandingan 60 hasil, bukan 60 training baru. Paket/run V5 terpisah dan notebook **29–32** telah dibuat dan seluruh 30 fit selesai pada pembaruan 4 Oktober. Notebook 28 tidak menjalankannya. [Panduan V5 sebagai referensi](PANDUAN_MANUAL_V5.md).

Tetapkan sebelum run: perubahan mean macro-F1 fold berpasangan sebagai ukuran utama, arah efek per seed dan recall dua kelompok RVOT sebagai pendamping. Prioritaskan kandidat bila mean perubahan positif, arah peningkatan muncul pada sedikitnya dua dari tiga seed, serta rata-rata recall kedua kelompok RVOT tidak menurun dibanding kontrol. Laporkan pula pooled OOF per seed, variasi antar-fold/seed, collapse, epoch dan biaya; trade-off tetap ditulis bila kriteria tidak selaras. Ini aturan eksplorasi development, belum bukti generalisasi atau uji signifikansi independen pada 15 fold/seed yang saling terkait. RNG sampling dapat menghasilkan pilihan berbeda antara K=4 dan K=8; jangan mengklaim empat window kontrol selalu menjadi subset persis kandidat.

Sesudah ablation stabil, baru tentukan kebutuhan tuning ulang, eksperimen GAN yang dikoreksi sesuai tujuan proposal, dan penguatan evaluasi/naskah. Lima puluh sembilan pasien test historis sudah pernah dilihat; tidak dipakai untuk memilih arm, seed atau ensemble berikutnya.

## Status fase dan artefak

- [x] V4 notebook 20–27 lengkap dan audit hasil/checkpoint selesai.
- [x] Notebook 28: audit input development 230/230 dan 12 pratinjau.
- [x] Verifikasi sumber raw, tabel kualitas dan ringkasan kandidat; analisis deskriptif hard/control tersedia.
- [x] Desain ablation K=4 versus K=8 ditentukan dari development.
- [x] Implementasi run/notebook ablation V5 terpisah dan persiapan pada artefak asli; 30 kontrol/scaler diverifikasi.
- [x] Smoke/pilot perangkat dan 30 fit ECG asli manual V5.
- [x] Analisis paired development V5: K8 tidak diprioritaskan, kontrol K4 tetap.
- [ ] GAN terkoreksi jika tetap dalam objektif, evaluasi diperkuat, dan revisi Bab III–V.

Notebook 28 tidak mengubah skor classifier. Hasil primer Hierarchical V4 tetap **CV macro-F1 0,6616; historical test seed 42 0,6198; mean lima seed 0,5421 ± 0,0845**. Angka 0,9 belum tercapai dan belum ada bukti bahwa skor saat ini merupakan batas maksimal. Sasaran publikasi tidak dapat dipastikan dari satu nilai metrik.

Analisis dapat direproduksi dari folder project tanpa training:

```bash
.venv/bin/python -m analysis_v4.analyze_input_audit
```

Output baru berada di `results/research_v4/20261002_patient_mil_hybrids_v4/development_input_audit_20261003/analysis_20261003_complete/`: `input_audit_analysis.json`, `verified_patient_inputs.csv`, `descriptive_matched_pairs.csv`, `training_window_coverage_hypothesis.csv`, `detector_review_flags.csv`, dan `windows_without_candidates_patients.csv`. Kode audit input dan notebook dengan output pengguna dipertahankan. [Fase setelah V4](FASE_SETELAH_V4_20261003.md) · [Analisis classifier V4](ANALISIS_HASIL_V4_20261003.md).
