# Menjalankan perbaikan berikutnya secara manual

**Status terbaru 2 Oktober 2026 setelah run manual:** notebook 12–15 menyelesaikan **630 CV, 20 final, 20 test**; semuanya lolos audit. **Notebook 18 selesai 7/7 sel, notebook 19 selesai 3/3 sel**, tanpa error tersimpan; SHAP Hierarchical V3 default 15 pasien telah diperiksa. **Notebook 17 kini juga selesai 5/5 sel**, seluruh 75 rekaman INCART tersedia dan hasil lolos audit: F1 deteksi beat 0,96862, recall V 92,79%. [Analisis INCART dan kegagalan](ANALISIS_BENCHMARK_INCART_20261002.md) · [analisis classifier/SHAP](ANALISIS_HASIL_V3_20261002.md). Tidak perlu mengulang notebook 12–19. Tahap berikutnya adalah protokol classifier baru pada development dan penulisan TA; implementasi/training lanjutan belum disiapkan.

**Model primer:** Hierarchical window_segment S04, LR 5e-4, batch 16, 9 epoch, seed 42. Notebook 19 mengambil pilihan tersebut otomatis, bukan memilih model dari test tertinggi. Notebook 16 tetap SHAP Encoder V2. Bagian berikut menjelaskan riwayat perbaikan flags dan cara resume bila dibutuhkan.

**Status audit setelah run manual 1 Oktober 2026:** persiapan notebook 11, pilot 12–14, seluruh ExtraTrees 15 dan SHAP 16 sudah selesai. V3 **150/630 CV**; sisa **480 CV neural**, final/test belum ada. Notebook 17 belum menjalankan benchmark karena flag False. [Analisis pilot dan audit aktual](ANALISIS_PILOT_V3_20261001.md).

**Perbaikan alur setelah run ulang 1 Oktober 2026:** run ulang masih memakai flag False dan melewati kasus pilot yang sudah selesai, sehingga tetap 150/630. Default notebook **12–14 sekarang `RUN_FULL_GRID=True`**. Notebook **15 sekarang `RUN_FINAL_AND_TEST=True`**, dengan pemeriksaan seluruh kasus CV sebelum seleksi/final/test. Jika belum lengkap, ia menampilkan jumlah tersisa dan melewati final/test. Empat uji routing/readiness lolos; paket training dan fingerprint V2/V3 tetap valid.

**Cara resume yang digunakan pada 1 Oktober:** buka ulang notebook 12, Restart Kernel lalu Run All. Sel pengaturan mencetak **`Mode: FULL GRID 180 fit`**; lanjutkan 13 → 14 → 15. Kini training, analisis/SHAP 18–19 dan benchmark 17 sudah lengkap. Default tetap disiapkan untuk resume dan kasus selesai dilewati. Training dijalankan manual oleh pengguna.

Output notebook 12–15 sebelum perubahan tersimpan di `arsip/notebooks_sebelum_fullgrid_20261001/`. Output pada notebook aktif dibersihkan agar tampilan run lama tidak dianggap sebagai hasil full grid baru; seluruh checkpoint/metrik/log training tetap tersedia.

## Urutan notebook

| Notebook | Isi | Yang perlu Anda lakukan |
|---|---|---|
| 11_Audit_Referensi_dan_Persiapan_V3 | Audit referensi, freeze protokol, siapkan kandidat development, quality flags dan visualisasi | Run All menggunakan kernel `.venv`. Tidak melatih classifier. |
| 12_CV_Encoder_V3 | S01 pada empat arm, kemudian seluruh grid Encoder | Default full grid True. Buka ulang, Restart Kernel & Run All; kasus selesai dilewati. |
| 13_CV_Patch_Transformer_V3 | Grid CNN patch transformer, termasuk koreksi tail patch | Default full grid True. Jalankan setelah 12. |
| 14_CV_Hierarchical_V3 | Grid Hierarchical | Default full grid True. Jalankan setelah 13. |
| 15_ExtraTrees_Seleksi_Final_dan_Laporan_V3 | 90 CV comparator, seleksi, 20 final, historical test, grafik | Default final/test True. Run All setelah 12–14; kesiapan seluruh CV wajib lengkap sebelum final/test. |
| 16_SHAP_Model_Final_V2 | Penjelasan checkpoint Encoder V2 yang sudah tersedia | Dapat dijalankan kapan saja, tidak harus menunggu V3. CPU default, resume per pasien. |
| 17_Benchmark_Detektor_INCART | Benchmark detektor dengan anotasi beat publik | Selesai 5/5 sel, seluruh 75 rekaman dan hasil terverifikasi. Baca tabel tersimpan/laporan; tidak perlu mengunduh atau menghitung ulang. |
| 18_Analisis_Hasil_Final_V3 | Audit 630 CV/20 final/20 test, parameter, bootstrap, class recall, seed dan grafik | Selesai 7/7 sel. Buka output untuk membaca hasil; tidak melatih classifier. |
| 19_SHAP_Model_Primer_V3 | SHAP model primer V3 yang dipilih CV, sekarang Hierarchical seed 42 | Selesai 3/3 sel; seluruh default 15 pasien terverifikasi. CPU, 30 background, nsamples 100; resume per pasien. Belum mencakup seluruh 59 pasien test. |

## Protokol yang dibekukan sebelum melihat CV V3

Run default `20261001_patient_r_candidates_v3`. Kohort tetap 230 pasien development dan 59 historical test. Tiga label tetap **LVOT**, **RVOT_cusp**, **RVOT_septal_freewall_other**. Pasien Summit/unknown tetap di luar endpoint tiga kelas, tanpa perubahan diagnosis.

| Arm | Representasi | Probabilitas sampling training |
|---|---|---|
| window_segment | 2,5 detik, HPF+DWT V2 | Sama untuk tiap segmen |
| window_patient | Sama | Sama untuk tiap pasien dalam ekspektasi, bobot segmen 1/n segmen pasien |
| r_candidate_segment | 0,5 detik di sekitar kandidat consensus | Sama untuk tiap crop |
| r_candidate_patient | Sama | Sama untuk tiap pasien dalam ekspektasi |

Semua arm memakai sampling dengan pengembalian dan inverse **patient** class-frequency loss dari training fold. Ini berbeda dari inverse **segment** class-frequency V2. Jumlah draw per epoch mengikuti jumlah window 2,5 detik pada training fold sehingga jumlah update sama untuk LR/batch yang sama. Validation memakai seluruh potongan asli yang tersedia, lalu probabilitas dirata-ratakan per pasien.

Candidate dibuat dari raw ECG dengan bandpass 5–25 Hz untuk deteksi derivative-energy, dua sadapan terkuat dari II/V2/V5, tolerance consensus 50 ms dan refractory 250 ms. Crop memakai waveform preprocessing V2, bukan sinyal bandpass detector. Panjang 200 ms sebelum dan 300 ms setelah adalah keputusan engineering yang ditetapkan, bukan panjang optimal yang sudah dibuktikan atau reproduksi penuh paper. Maksimal 64 crop dipilih merata sepanjang rekaman. Pasien tanpa candidate lengkap memperoleh window fallback 500 ms yang ditandai; tidak dibuang. Quality flag tidak digunakan untuk mengubah label atau mengecualikan pasien.

**R/QRS candidate bukan anotasi PVC.** Label ablation-site melekat pada pasien, sedangkan isi beat dapat mencampurkan sinus/PVC/VT. Arm tersebut merupakan pendekatan dengan label tingkat pasien. Kesepakatan sadapan tinggi tidak membuktikan bahwa semua PVC terdeteksi. Preview 12 pasien mencatat satu kasus dengan agreement 0,138; diperlukan benchmark/analisis kegagalan, bukan klaim detector sudah akurat.

Untuk tiga neural model tetap digunakan LR **[1e-4, 5e-4, 1e-3]**, batch **[16, 32, 64]**, semua 9 kombinasi × 5 fold. Pilot S01 termasuk grid, tidak mengulang fit yang selesai ketika grid lengkap berjalan. Total neural CV **540 fit**; comparator ExtraTrees memakai 9 kombinasi max_features/min_samples_leaf × 2 representasi × 5 fold = **90 fit**. Total **630 fit**. Tabel pilot belum cukup untuk memilih final.

Encoder/Hierarchical memakai arsitektur V2. CNN patch transformer V3 menyimpan bagian feature sequence terakhir melalui padding dan pooling sesuai panjang valid. V2 memotong sisa urutan; contohnya 62 timestep dipotong menjadi 48 untuk patch 16. Jangan mengatribusikan perbedaan Patch V2–V3 hanya pada patient sampling; gunakan pasangan arm V3 dengan konfigurasi yang sama.

ExtraTrees menggunakan 500 pohon dengan max_depth tetap 12, 132 fitur waveform deskriptif/bandpower. Ini baseline komputasional yang baru, bukan reproduksi pengukuran klinis Zheng. Bobot training menghitung pasien dan kelas hanya pada training fold. Tanpa expert, fitur tidak diberi nama diagnosis klinis yang belum dibuktikan.

Pemilihan per model dari mean macro-F1 CV; tie-break SD, arm, scenario. Model neural utama dipilih dari CV dan comparator disajikan terpisah. Final tiap model dilatih pada development penuh, epoch median best epoch CV, seed 42–46. Seed **42** tetap hasil primer; lima seed tidak dipakai untuk mencari seed test terbaik. Sebelum semua final lengkap, pembacaan/evaluasi test melalui endpoint V3 terkunci. Test historis tidak pernah dipakai menentukan threshold, LR, batch, epoch atau arm.

## Endpoint terminal bila ingin menjalankan bagian tertentu

Jalankan dari root project menggunakan `.venv/bin/python`:

```bash
.venv/bin/python -u -m research_v3 prepare
.venv/bin/python -u -m research_v3 cv --model encoder --scenario S01 --device mps
.venv/bin/python -u -m research_v3 cv --model encoder --arm window_patient --device mps
.venv/bin/python -u -m research_v3 cv --model patchtst --device mps
.venv/bin/python -u -m research_v3 cv --model hierarchical --device mps
.venv/bin/python -u -m research_v3 classical --device cpu
.venv/bin/python -u -m research_v3 status
```

Perintah kedua adalah pilot 20 fit. Perintah ketiga hanya 45 fit untuk satu arm Encoder; arm lain tetap wajib diselesaikan sebelum seleksi. Untuk seluruh 180 fit Encoder, hapus opsi `--arm`. Tambahkan `--fold 1` dan `--scenario S01` jika ingin satu kasus saja. Log stream memperlihatkan epoch/fold; tekan Interrupt pada notebook untuk berhenti. Ulang sel/perintah yang sama untuk melanjutkan dari checkpoint epoch terakhir. CPU/MPS resume harus pada jenis perangkat yang sama.

Setelah status menunjukkan 630/630:

```bash
.venv/bin/python -u -m research_v3 select
.venv/bin/python -u -m research_v3 final --model encoder --device mps
.venv/bin/python -u -m research_v3 final --model patchtst --device mps
.venv/bin/python -u -m research_v3 final --model hierarchical --device mps
.venv/bin/python -u -m research_v3 final --model extratrees --device cpu
.venv/bin/python -u -m research_v3 test --device mps
.venv/bin/python -u -m research_v3 report
```

CV menggunakan 5 fold hanya pada development. Test 59 pasien dievaluasi tanpa k-fold. Lima seed ialah lima model final; checkpoint primer tetap satu model, tidak ada ensemble otomatis.

## Hasil dan interpretasi

Artefak baru di `results/research_v3/<run-id>/`, log `manual_logs/`, QC `representations/r_candidate/data/development/candidate_quality.csv`, CV per arm/model/scenario/fold, final, test dan reports. Cache window V2 dipakai sebagai read-only, bukan disalin. Candidate shards dan mmap menghindari memuat seluruh dataset sekaligus; program memeriksa ruang disk sebelum menulis. Jangan menjalankan beberapa fit bersamaan pada MPS.

Grafik report membandingkan best-grid CV (optimistic akibat seleksi), confusion matrix primer dan bootstrap pasien untuk historical test. Untuk mengisolasi patient sampling, bandingkan **scenario dan fold yang sama** antara arm, bukan hanya dua pemenang grid. `cv_folds.csv` menyimpan data pasangan tersebut. SD 5 fold bukan CI dari lima cohort independen.

SHAP V2 memakai satu segmen tengah untuk setiap pasien background development terpilih dan 5 pasien/class pada default explain set (total15). SHAP target ialah true-class **segment logit**. Background tidak mencakup test. Hasil resume berdasarkan checkpoint/code/parameter/version hash. Grafik waktu memakai 2000 Hz. Lead importance/kurva waktu belum menjadi interpretasi klinis tervalidasi atau penjelasan lengkap mean probability pasien.

## INCART tanpa anotasi baru dari ahli

Notebook 17 memerlukan **wfdb==4.3.1** dan soundfile/cffi/pycparser, yang belum dipasang pada environment saat kode disiapkan. Cell instalasi manual menggunakan target terpisah `.ecgqa/` dan `--no-deps` agar tidak memperbarui package training lama. NumPy/Pandas/SciPy/Matplotlib/requests project digunakan sebagai dependensi yang sudah tersedia. Sel instalasi mencatat package yang diunduh; benchmark mencatat versi WFDB. Jika library tidak tersedia/kompatibel, sel akan menunjukkan error yang perlu diperbaiki sebelum benchmark.

Pada penggunaan pertama, ubah sel pengaturan notebook 17 menjadi berikut, lalu **Restart Kernel → Run All**:

```python
INSTALL_WFDB = True
DOWNLOAD_DATA = True
RUN_BENCHMARK = True
```

**Perbaikan alur 2 Oktober setelah run ulang masih terlewati:** nilai di atas sekarang sudah menjadi pengaturan notebook 17. Output lama disimpan utuh di `arsip/notebooks_sebelum_benchmark_20261002/17_Benchmark_Detektor_INCART.ipynb`; output sel yang melewati proses pada notebook aktif dibersihkan. Jika notebook sudah terbuka di editor, tutup dan buka ulang agar memakai file terbaru. Saat flag run sengaja False, sel benchmark sekarang menampilkan error yang jelas, sehingga tidak terlihat seperti penyelesaian benchmark. Ini perubahan notebook saja; kode/protokol training dan detector tetap dibekukan.

**Run sesudah perbaikan sudah selesai:** dependency QA tersedia, data 75 rekaman lengkap, `INCART report:` dan tabel hasil tersimpan. Hasil berada di `results/detector_benchmark/incart/848d6db41c5d/`; analisis/audit/grafik di `analysis_20261002/`. Output sebelum perbaikan tetap arsip historis. Setelah dependency dan data tersedia, install/download dapat dikembalikan ke False bila sengaja mengulang; `RUN_BENCHMARK` harus tetap True saat ingin menghitung benchmark. Keberhasilan sel saja belum berarti benchmark selesai; periksa tabel hasil dan artefak.

`DOWNLOAD_DATA=True` mengunduh file publik INCART; seluruh 75 rekaman membutuhkan sekitar 0,8 GB waveform ditambah metadata. Default menilai seluruh75 rekaman, tanpa pencarian threshold. Sampling mengikuti header 257 Hz, dan waktu filter/jendela tetap dalam satuan detik. Kandidat vs anotasi dinilai dengan matching satu-ke-satu ±100 ms, TP/FP/FN, precision/recall/F1, timing error, serta recall deteksi pada beat beranotasi V. Tidak ada klasifikasi PVC yang dilatih atau precision diagnosis PVC.

Benchmark ini menguji ketepatan deteksi beat pada dataset berbeda, **bukan validasi eksternal classifier tiga kelompok asal PVC**. Anotasi tepi yang tidak bisa menghasilkan crop lengkap dikecualikan dengan aturan yang sama untuk referensi/prediksi dan jumlahnya dilaporkan. Jika hasil detector rendah, laporkan kegagalannya; revisi algoritme harus membuat versi/run baru dan benchmark yang telah diamati dianggap data development untuk revisi itu.

Audit dan visualisasi hasil yang sudah selesai dapat dibaca atau diulang tanpa instalasi, download atau menjalankan detector:

```bash
.venv/bin/python -u -m analysis_v3.analyze_incart --benchmark results/detector_benchmark/incart/848d6db41c5d
```

Endpoint ini sudah dijalankan dan lolos: 225 hash, 75 jumlah reference dari anotasi, JSON/CSV/summary serta identitas V2/V3. CI memakai 32 grup sumber Holter. [Laporan lengkap](ANALISIS_BENCHMARK_INCART_20261002.md) memisahkan F1 semua beat, recall V dan hasil classifier.

## Status ilmiah

Grid penuh dan test V3 sudah tersedia. Kandidat 0,5 detik tetap lebih rendah daripada window 2,5 detik secara mean berpasangan. Sampling pasien memberi kenaikan CV rata-rata Encoder/Patch, tetapi stabilitas antar-seed neural dan pemisahan subkelompok RVOT masih bermasalah. Tidak ada jaminan 0,9 atau Q1. Label anatomi sumber dapat dipertahankan tanpa ahli baru, tetapi validasi klinis crop/interpretasi belum tersedia. GAN terkoreksi dan ablation GAN belum diimplementasikan dalam V3 ini; tetap pending bila wajib pada objektif TA. Nested/repeated CV, validasi pasien baru dengan label anatomi dan kontribusi ilmiah masih perlu dikerjakan.

Endpoint analisis/SHAP setelah final lengkap:

```bash
.venv/bin/python -m analysis_v3.analyze_final
.venv/bin/python -u -m analysis_v3.explain_final --device cpu --background 30 --patients 15 --nsamples 100
```

Kedua endpoint sudah dijalankan manual melalui notebook 18–19. Artefak analisis lengkap di `analysis_20261002_complete/`; SHAP V3 di `explain_primary_v3/729aa202d61f/` dalam run V3. Signature SHAP mencakup checkpoint/scaler/kode/versi/config/sampel. SHAP target logit satu segmen, bukan seluruh mean probability pasien, dan tidak menjadi validasi klinis.

Audit artefak SHAP dapat diulang tanpa training atau menghitung ulang SHAP:

```bash
.venv/bin/python -u -m analysis_v3.audit_shap --explanation results/research_v3/20261001_patient_r_candidates_v3/explain_primary_v3/729aa202d61f
```

Audit ini sudah lolos: delapan hash sesuai, background/test tanpa pasien beririsan, array dan ranking cocok. Ia menjalankan forward CPU pada 15 segmen untuk memberikan konteks prediksi; `shap_v3_patient_context.csv` membedakan prediksi segmen dari agregasi pasien. Hasil audit ini tidak menilai akurasi klinis SHAP.

Audit referensi: [AUDIT_DAFTAR_PUSTAKA_20261001.md](AUDIT_DAFTAR_PUSTAKA_20261001.md). Pembaruan metodologi: [ADDENDUM_METODOLOGI_V3_TANPA_AHLI_ECG.md](ADDENDUM_METODOLOGI_V3_TANPA_AHLI_ECG.md).
