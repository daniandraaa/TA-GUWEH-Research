# Fase setelah V4 — audit masukan lalu ablation terkontrol

**Pembaruan fase C — 6 Oktober 2026:** perbandingan classifier V8 selesai, **120 fit/notebook 41–45** lolos audit. GAN belum membantu secara konsisten dan recall RVOT other turun. [Analisis V8](ANALISIS_HASIL_V8_20261006.md). [Fase setelah V8](FASE_SETELAH_V8_20261006.md) memprioritaskan stabilitas epoch dan representasi real-only; tidak mengulang GAN/classifier yang telah lengkap atau membuka test.

**Fase C classifier kini diimplementasikan pada V8 — 5 Oktober 2026:** GAN V7 tetap lengkap/PASS. Mulai **41 → 42**, lalu **43 → 44 → 45** untuk perbandingan K4 real-only/real augmentation/GAN augmentation. Ada 30 selector inner dan 90 pembanding outer; training data asli tetap manual. [Panduan V8](PANDUAN_MANUAL_V8.md).

**Fase C aktif — 5 Oktober 2026:** GAN/QC V7 selesai, notebook 37–40 tanpa error dan **5/5 fold PASS** terverifikasi dari waveform native. [Analisis V7](ANALISIS_HASIL_V7_20261005.md). Fase C berikutnya adalah classifier K4 real-only/kontrol augmentasi real/real+GAN pada run baru; endpoint belum tersedia. [Rancangan](FASE_CLASSIFIER_SETELAH_V7_20261005.md). Training GAN tidak perlu diulang; efek classifier masih harus diuji.

**Fase A–B selesai.** Notebook 28 lengkap pada 230 development/12 pratinjau; notebook 29–32 dan **30/30 training V5** telah selesai. K8 menurunkan mean development macro-F1 kedua model sehingga **K4 dipertahankan**. [Analisis lengkap V5](ANALISIS_HASIL_V5_20261004.md). Model primer CV V4 Hierarchical tetap historical test seed42 0,6198; mean lima seed 0,5421 ± 0,0845. V5 tidak melatih final atau membaca sinyal test.

Sumber dataset menggunakan input di sekitar puncak R saat PVC/VT, dengan sampling 2.000 Hz dan ukuran window dipilih lewat validation; proyek saat ini memakai window 2,5 detik tanpa anotasi waktu PVC lokal. Ini perbedaan representasi yang perlu diperiksa, bukan bukti bahwa semua window proyek salah. Rujukan: [Zheng dkk., 2022, bagian ECG Measurement Protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC8962834/#sec2). Label asal dari sumber dipertahankan dan endpoint tetap LVOT / RVOT cusp / RVOT septal-freewall-other. Crop 0,5 detik berbasis kandidat pada V3 pernah diuji dan belum menang; tahap baru tidak sekadar mengulang kandidat pendek tersebut.

| Fase | Pekerjaan | Keluaran dan syarat lanjut | Status |
|---|---|---|---|
| A | Audit masukan development, notebook 28 | 230 pasien, daftar hard/control, integrity/window/QRS proxy, pratinjau 12 sadapan | Selesai 230/230 dan 12 pratinjau; analisis tambahan lulus |
| B | Ablation K=4 versus K=8, batch pasien tetap delapan | Perbandingan berpasangan pada fold/seed yang sama, variasi training dan recall kedua RVOT | Selesai V5/notebook 29–32: smoke, 30 fit dan 30 kontrol diaudit; K4 tetap |
| C | Lengkapi metode proposal: baseline/tuning/GAN terkoreksi | Mutu sintetis dan perbandingan dengan recipe/anggaran yang sebanding | GAN V7/notebook 37–40 selesai dan 5/5 fold PASS; endpoint/manfaat classifier masih tahap berikut |
| D | Evaluasi diperkuat dan naskah | Nested/outer patient evaluation bila layak, ketidakpastian, hasil negatif, Bab III–V | Penulisan dapat dimulai; evaluasi tambahan belum selesai |

## A. Audit yang sudah selesai — referensi alur

Notebook: `28_Audit_Input_Development_dan_Window_ECG.ipynb`. Pilih kernel `.venv` project, kemudian jalankan sel satu per satu:

1. **Setup dan prepare.** Memvalidasi fingerprint V4, membaca selected OOF development dan membuat daftar pemeriksaan. Pasien hard mempunyai setidaknya empat model salah; kontrol mempunyai paling banyak satu model salah. Prioritas matching: kelas/sublokasi sama, kemudian jumlah window terdekat. Fallback dan kontrol yang dipakai ulang dicatat. Seluruh 230 development tetap diperiksa, termasuk kelompok intermediate.
2. **Pilot empat pasien.** Membaca raw/cache, memeriksa window overlap, mengukur amplitudo/variasi numerik dan menghitung kandidat konsensus dengan detector yang sudah digunakan pada V3. Ini pemeriksaan data pada CPU, tanpa classifier training.
3. **Audit lengkap.** Ulang sel untuk resume pasien yang belum selesai. Status harus `completed_patients: 230` dan `complete: True` sebelum membuat keputusan eksperimen berikutnya.
4. **Pratinjau 12 pasien.** Raw lima detik pertama dibandingkan dengan window tengah cache sebelum scaler fold. Dua kolom tidak selalu berasal dari waktu sama. Marker merah adalah kandidat konsensus QRS, bukan anotasi PVC tervalidasi.
5. **Ringkasan.** Periksa jumlah/durasi window, tail rekaman yang tidak masuk cache, kesesuaian overlap, kanal numerik konstan/identik, amplitudo dan derivative RMS, agreement detector, serta window tanpa kandidat lengkap.

Output: `results/research_v4/20261002_patient_mil_hybrids_v4/development_input_audit_20261003/`. File utama: `hard_control_pairs.csv`, `development_review_inventory.csv`, `patient_signal_quality.csv`, `quality_by_review_group.csv`, `status.json` dan `waveform_previews/`. Hash raw, cache, detector dan kode audit dicatat; versi/masukan yang berubah ditolak saat resume. Jangan mengedit inventory untuk memasukkan pasien test. Audit hanya membaca raw/cache **development**; label, pasien, cache training dan checkpoint V4 tidak diubah.

Statistik sinyal memakai unit file/cache, belum diasumsikan mV. Rekonstruksi dari window menghitung sampel yang tertutup hanya sekali supaya overlap tidak menggandakan statistik amplitudo. Kandidat lengkap di tepi rekaman mengikuti batas crop detector 0,2 s sebelum dan 0,3 s sesudah pusat; batas ini bukan penentuan durasi PVC. Audit melepaskan cap 64 kandidat agar hitungan tidak turun hanya karena batas penyimpanan V3.

Interpretasi penting: window tanpa kandidat bisa menunjukkan kegagalan detector; kandidat yang ditemukan juga bisa sinus atau artefak. Kedua hal itu tidak membuktikan keberadaan/ketiadaan PVC. Flag teknis tidak digunakan untuk membuang pasien, mengubah label, memilih seed test, atau mengklaim validasi klinis. Pairs hard/control dan kelompok error dipilih sesudah OOF diamati, jadi ringkasan bersifat deskriptif; pasangan yang memakai kontrol sama tidak diperlakukan sebagai pasien independen.

Alternatif terminal dari folder project:

```bash
.venv/bin/python -m analysis_v4.input_audit prepare
.venv/bin/python -m analysis_v4.input_audit pilot
.venv/bin/python -m analysis_v4.input_audit scan
.venv/bin/python -m analysis_v4.input_audit review --count 12
```

## B. Ablation pertama selesai: ukuran bag training

**Keputusan 4 Oktober 2026:** K8 tidak diteruskan pada recipe ini. Mean fold/seed macro-F1 Hierarchical turun 0,0158, Encoder turun 0,0107; recall RVOT septal/freewall/other turun 10,48 dan 5,24 poin persentase. K8 tidak memenuhi aturan prioritas yang tercantum di bawah. Tidak perlu mengulang training V5 atau membuka test untuk memilih ulang. Nilai ini belum menjadi batas maksimal performa. Rincian di [analisis V5](ANALISIS_HASIL_V5_20261004.md); paragraf desain berikut dipertahankan sebagai jejak keputusan sebelum training.

Mulai pilot dari **Hierarchical dan Encoder**, dua model dengan CV V4 tertinggi. Perubahan pertama adalah maksimal window per pasien saat training **K=4 → K=8**. Ada 193/230 pasien mempunyai lebih dari empat window; median delapan. Pada K=4 median proporsi window terambil per kunjungan adalah 50%, sedangkan validation memakai semua window. Sampling berbeda tiap epoch dan overlap bukan informasi independen, sehingga ini hipotesis yang perlu diuji, bukan penyebab skor rendah yang sudah terbukti.

Pertahankan manifest pasien, tiga kelas, fold, seed 42/43/44, preprocessing, scaler, arsitektur/pooling, loss, optimizer, scheduler dan LR terpilih. **Batch pasien tetap delapan**, sehingga kapasitas segmen berubah 32 → 64 sebagai konsekuensi K bertambah. Kapasitas tetap 32 akan menurunkan batch pasien menjadi empat dan mencampur dua perubahan. Jalankan smoke perangkat dan catat biaya; jangan otomatis menurunkan batch pasien jika memori tidak cukup. Runner `notebooks/_v5_runner.py`, paket `research_v5/` dan notebook 29–32 telah disiapkan pada run V5 terpisah.

Anggaran pertama: **dua model × lima fold × tiga seed = 30 fit baru** untuk K=8. Tiga puluh hasil kontrol dapat merujuk selected stability V4 **bila seluruh recipe tetap identik**, dengan provenance reuse; total 60 hasil pembanding. Ini satu ablation development, belum full grid baru atau estimasi nested generalisasi. Perubahan lain seperti normalisasi, kompresi temporal dan pretraining belum dipilih untuk run pertama.

Laporkan perubahan mean macro-F1 fold berpasangan, arah efek per seed, recall tiap RVOT, pooled OOF, kasus collapse, epoch dan biaya komputasi. Prioritaskan kandidat bila mean perubahan positif, peningkatan muncul pada setidaknya dua dari tiga seed dan rata-rata recall kedua RVOT tidak turun; bila hasilnya bertentangan, tulis trade-off dan jangan mengklaim perbaikan konsisten. Fold/seed yang saling terkait tidak diperlakukan sebagai 15 sampel independen. Sesudah recipe ditetapkan, perluas ke tiga arsitektur proposal dan grid LR `[1e-4, 5e-4, 1e-3]` × batch `[16, 32, 64]` bila diperlukan untuk perbandingan inti, dengan arti batch pasien/segmen dinyatakan. Perubahan recipe/model memakai paket/run baru; V4 dipertahankan. [Desain rinci dan batas interpretasi](ANALISIS_AUDIT_INPUT_V4_20261003.md#desain-ablation-pertama).

Pretraining dapat diuji bila diperlukan untuk kontribusi CNN–Transformer dan *fine-tuning* dari bobot pretrained yang sebenarnya. V4 saat ini merupakan training dari awal dengan tuning hyperparameter. Pretraining hanya memakai training portion yang sesuai; representasi, sumber bobot dan biaya harus dicatat. Ini kandidat lanjutan setelah A/B, belum perbaikan yang terbukti dan belum diimplementasikan pada notebook 28.

## C–D. Proposal dan kekuatan evaluasi

GAN terkoreksi tetap pekerjaan terbuka bila augmentasi GAN masih tujuan proposal. Kerjakan pilot mutu dahulu (amplitudo, frekuensi, keragaman, hubungan sadapan dan memorisasi), lalu classifier real-only versus real+GAN pada fold/seed/recipe yang sama. Generator/scaler hanya fit pada training fold; validation/test tetap data asli. GAN V1 dengan preprocessing/label historis tidak dipakai sebagai generator untuk run baru.

Penguatan evaluasi mengikuti recipe yang sudah stabil: inner selection dan outer patient scoring dipisah, epoch tidak dipilih dari outer validation. Kohort eksternal dengan label asal yang kompatibel dapat ditambahkan bila tersedia; INCART merupakan benchmark detector, bukan validasi classifier tiga kelas. Lima seed selalu dilaporkan dan 59 pasien test tetap test historis yang sudah diamati. Mereka tidak digunakan untuk memilih arm atau komposisi ensemble berikutnya.

Bab III dapat diperbarui sekarang dengan prosedur V3/V4; Bab IV memakai tabel audit, variasi seed, kebingungan RVOT dan hasil negatif kedua model tambahan. Kesimpulan tentang 0,9/Q1 menunggu bukti metode dan evaluasi berikutnya. Tidak ada jaminan bahwa ablation, pretraining atau GAN akan mencapai angka tertentu.


**Verifikasi audit input:** empat uji sintetis persiapan lulus (matching kelas/reuse, overlap/corruption, gap/nonfinite, batas window). Notebook 28 selesai 230/230 development dan 12 pratinjau; seluruh sel kode memiliki execution count dan tidak menyimpan error. Analisis tambahan mencocokkan 230 hash raw dengan manifest preprocessing asli, seluruh kolom kualitas dengan JSON pasien dan hitungan kandidat dengan window. Matching menghasilkan 37 pasangan dengan 32 kontrol unik: 36 cocok kelas/sublokasi, satu kelas saja, lima memakai ulang kontrol. Notebook 28 tidak melatih classifier atau mengubah label/pasien. **Verifikasi V5:** tujuh uji sintetis persiapan, kedua smoke MPS pengguna, 30 training baru, 30 kontrol, 300 hash hasil dan 3.098 artefak V4 lulus; audit tambahan hanya membaca artefak dan seluruh notebook/laporan asli dipertahankan.
