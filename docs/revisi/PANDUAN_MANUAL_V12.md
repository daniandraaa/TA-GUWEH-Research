# V12 — identitas beat sebelum revisi masukan lokalisasi

**Pembaruan hasil manual, 7 Oktober 2026:** 62–67 selesai; 40 fit CV dan satu deployment terverifikasi. Mean fold F1 RR 0,7897, tetapi MIT-BIH F1 **0,3998/gate FAIL**; 68 melewati ekspor. **Tidak perlu mengulang run ini atau memaksa ekspor.** Baca [diagnosis dan prioritas berikut](ANALISIS_HASIL_V12_20261007.md). Petunjuk di bawah dipertahankan sebagai referensi pelaksanaan V12.

**Mulai dari notebook 62.** Kernel `.venv` project, jalankan sel satu per satu dengan **Shift+Enter**. Semua endpoint V12 memakai CPU; satu proses pada satu waktu. Training/inference data asli dilakukan pengguna secara manual. Tidak perlu mengulang notebook 00–61 yang telah selesai.

Run baru: `20261007_beat_identity_v12`. Kode: `research_v12/`; runner: `notebooks/_v12_runner.py`; keluaran: `results/research_v12/20261007_beat_identity_v12/`. Tidak bergantung pada cache encoder V10 yang hilang.

## Tujuan dan status

V12 menguji **jenis denyut** sebelum membuat masukan lokalisasi lebih spesifik. Tiga output ialah:

- `normal_N`: beat beranotasi N.
- `PVC_V_or_r`: V atau r sesuai simbol sumber.
- `other_known_beat`: beat lain yang terdefinisi, termasuk supraventrikular, bundle-branch, escape, fusion atau paced.

Q/unclassifiable tidak dijadikan normal atau label training; jumlahnya dilaporkan. Marker irama/non-beat tidak menjadi contoh beat. Fusion berada dalam kelompok other untuk tugas pencocokan simbol, bukan klaim bahwa fusion tidak mempunyai komponen ventrikular. **Normal-N lokal belum dianggap sinus terverifikasi.**

Kelompok asal anatomi tetap LVOT/RVOT cusp/RVOT septal-free-wall-other seperti sebelumnya. V12 belum melatih classifier asal anatomi dan tidak membuka historical origin test. Ini baseline engineering identitas beat berbasis fitur + logistic regression L2, bukan reproduksi uPVC-Net atau fine-tuning CNN–Transformer. Keberhasilan V12 akan menyediakan kandidat masukan bagi fase CNN–Transformer berikutnya.

Implementasi/notebook sudah tersedia. Sebelum diserahkan, metadata/header/anotasi INCART diperiksa tanpa membaca waveform: **75 rekaman, 32 grup sumber**, split outer/inner dapat dibentuk. Reference eligible berkelas: N **150.346**, V/r **20.007**, other **5.475**. Angka ini mengikuti eligibility crop 0,8 detik V12 dan belum jumlah kandidat training yang matched. Belum ada hasil classifier ECG asli baru.

## Urutan notebook

| Urutan | Notebook | Yang dilakukan | Keluaran/syarat |
|---|---|---|---|
| 62 | [Persiapan identitas beat](../../notebooks/62_V12_Persiapan_Identitas_Beat.ipynb) | Audit metadata, hash dataset, split grup dan smoke sintetis | Persiapan lengkap; smoke lulus |
| 63 | [Ekstraksi fitur INCART](../../notebooks/63_V12_Ekstraksi_Fitur_Beat_INCART.ipynb) | Detector kandidat, pencocokan reference, fitur per beat/lead dan QA | 75/75 shard lengkap; QA deteksi tersedia |
| 64 | [Pilot fold 1](../../notebooks/64_V12_Pilot_Identitas_Beat_Fold1.ipynb) | Dua arm, tiga fit C inner dan satu outer per arm | **8 fit**, termasuk dalam full CV |
| 65 | [CV lima fold](../../notebooks/65_V12_CV_Identitas_Beat_Lima_Fold.ipynb) | Menggunakan kembali pilot, melengkapi fold 2–5 | **40 fit total**, tambahan 32 sesudah pilot |
| 66 | [Analisis dan deployment](../../notebooks/66_V12_Analisis_OOF_dan_Deployment_Detektor.ipynb) | Audit semua metrik evaluasi dari prediksi tersimpan, visualisasi; satu fit deployment dari INCART | OOF/audit/plots dan satu model jenis beat |
| 67 | [Evaluasi MIT-BIH](../../notebooks/67_V12_Evaluasi_Eksternal_MITBIH.ipynb) | Download/extract manual jika perlu; inference model tetap dan audit/CI | Metrik lintas dataset dan gate PASS/FAIL; tanpa fit/tuning |
| 68 | [Ekspor kandidat lokal](../../notebooks/68_V12_Ekspor_Kandidat_Lokal_dengan_Mask.ipynb) | Hanya setelah 67 PASS: inference 230 development dan crop 12 lead | Kandidat, mask, fallback dan coverage; tanpa training origin |

**Langkah pertama cukup 62 → 63 → 64**, kemudian periksa apakah delapan fit pilot selesai. Hasil pilot rendah tetap dicatat; full CV memberi gambaran lebih baik. Kirim hasil 66/67 setelah selesai, atau log lengkap bila ada error teknis. Jika 67 FAIL, 68 akan melewati ekspor dan menampilkan alasan; analisis kegagalan diperlukan sebelum eksperimen baru.

Pada notebook 67, `DOWNLOAD_MITBIH=True` berarti sel tersebut boleh mengambil file dataset resmi. Setelah download lengkap boleh diubah menjadi False. Tidak ada pemasangan dependency atau perubahan `.venv`; WFDB menggunakan target QA `.ecgqa` yang sudah tersedia. Data mentah MIT-BIH sekitar 104 MB. Semua tahap menjaga cadangan **3 GiB** dan memperkirakan kebutuhan tambahan; bila guard berhenti, kosongkan file non-riset atau sediakan ruang penyimpanan tanpa menghapus hasil eksperimen.

## Desain yang dibekukan sebelum training

**Grouping:** [INCART resmi](https://physionet.org/content/incartdb/1.0.0/) menjelaskan 75 cuplikan dari 32 Holter dan patient number pada header. Semua cuplikan satu nomor disimpan dalam grup yang sama. Outer fivefold dan satu inner grouped split per outer fold dibentuk dari anotasi, tidak dari skor model. INCART merupakan development setelah benchmark lamanya telah dibaca; hasil V12 di sana tidak diklaim sebagai external test baru.

**Eksternal:** [MIT-BIH resmi](https://physionet.org/content/mitdb/1.0.0/) memiliki 48 rekaman dari 47 subjek. [Direktori sumber](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm) menyebut 201/202 dari orang yang sama, sehingga bootstrap memakai satu grup untuk keduanya. Data berasal dari kanal Holter yang dapat berbeda dari lead standar lokal. Semua rekaman, termasuk paced cases, dipertahankan; tidak ada pilihan C/threshold dari MIT-BIH. Sumber ini memvalidasi jenis beat, bukan origin.

Endpoint eksternal memverifikasi seluruh CV, split dan deployment sebelum membuka/download anotasi MIT-BIH. Setelah snapshot eksternal dibuat, fitting baru pada run yang sama dibekukan; resume hanya memakai fit yang telah lengkap dan cocok. Model/threshold yang dinilai tetap berasal dari INCART.

**Masukan:** kanal II/V1 INCART/lokal dan MLII/V1 MIT-BIH bila tersedia. Nama lead sebenarnya dipertahankan; bila hanya satu kanal tersedia, tidak membuat kanal kedua palsu. Probabilitas dirata-ratakan antarlead untuk satu keputusan per beat. Filter fitur 0,5–40 Hz, crop 0,3 s sebelum + 0,5 s sesudah pusat detektor. Fitur: 64 titik bentuk hasil resampling dan delapan deskriptor energi/polaritas/amplitudo relatif (**72 fitur/lead**). Arm RR menambah tiga fitur (**75 fitur/lead**). Lebar energi bukan durasi QRS klinis/LUDB yang tervalidasi.

**RR dan matching:** RR dihitung dari kandidat detector, termasuk tetangga sebelum/sesudah, sesuai penggunaan offline rekaman lengkap. Training menggunakan pusat kandidat yang matched ke anotasi dengan tolerance 100 ms, bukan pusat anotasi oracle. Candidate detector adalah adaptasi heuristik sebelumnya untuk kanal yang tersedia; bukan detector V3 yang identik. Crop lengkap dan eksklusi tepi dicatat. Posisi anotasi INCART kadang kurang selaras menurut sumber, sehingga tolerance dan batas tersebut ikut membatasi interpretasi.

**Pembobotan:** maksimum 600 beat per kelas yang hadir per grup training, deterministik; evaluation tidak memakai cap. Bobot memberi kontribusi sama setiap grup, lalu kelas yang hadir, beat dan kanalnya. Scaler/imputer hanya fit dari training yang dipilih. Cap merupakan penghematan training yang dinyatakan, bukan eksklusi pasien atau pembatasan beat evaluasi.

**Grid:** C `[0.01, 0.1, 1.0]` × threshold PVC `[0.5, 0.7, 0.9]` = sembilan kombinasi nilai evaluasi per arm/fold, tetapi cukup **tiga fit inner** karena threshold tidak mengubah bobot. Ini grid head L2 dan ambang jenis beat. Grid learning rate/batch CNN–Transformer belum dijalankan pada V12.

**Seleksi:** inner end-to-end PVC F1, lalu precision, C lebih kecil dan threshold lebih tinggi sebagai tie-break. Outer refit memakai seluruh outer training; outer tidak menjadi early stopping atau pemilih parameter. Sesudah semua fold, pilih arm berdasarkan mean outer PVC F1; C/threshold deployment memakai joint mode pilihan inner arm tersebut, dengan tie-break tetap. Satu fit deployment memakai seluruh INCART. Mean CV arm yang dipilih tetap merupakan hasil development/seleksi; evaluasi eksternal menilai model yang dibekukan.

**Skor probabilitas belum dikalibrasi.** Bobot kelas/grup dan soft voting dapat memengaruhi makna skor. Brier pada matched reference dilaporkan sebagai diagnosis; score tinggi tidak menjadi kepastian klinis.

## Cara membaca hasil

Metrik utama adalah **end-to-end PVC precision/recall/F1**:

- Semua reference PVC eligible menjadi denominator recall, termasuk yang gagal dideteksi.
- Prediksi PVC pada kandidat yang tidak matched reference dihitung false positive.
- Reference Q/unclassifiable dipisahkan; jumlah kandidat Q dan prediksi PVC/normal-N pada Q dilaporkan. Precision mengabaikan reference Q, sehingga belum memvalidasi keputusan pada beat yang tidak mempunyai label terdefinisi.
- Matched-known macro-F1/ROC-AUC/PR-AUC/Brier merupakan metrik diagnosis classifier setelah deteksi. Nama tersebut tidak boleh disingkat menjadi skor keseluruhan pipeline.
- OOF mempertahankan satu keputusan setiap kandidat, bukan satu keputusan per lead. Bootstrap mengambil grup sumber lengkap; interval bersyarat pada prediksi tetap dan tidak mencakup retraining/seluruh seleksi metode.

Gate pada 67 ditetapkan sebagai **aturan rekayasa usulan ini**, bukan standar klinis: PVC precision/recall/F1 masing-masing ≥0,80; precision normal-N terpilih ≥0,80 pada threshold normal-N 0,90; sedikitnya **25% seluruh reference normal-N terpilih benar**. Pilihan normal tidak boleh kosong. Gate tidak boleh diturunkan setelah melihat kegagalan untuk membuka ekspor lokal. Setelah MIT-BIH digunakan untuk mendesain revisi, statusnya berubah menjadi sumber yang telah diamati; eksperimen baru perlu dinyatakan demikian dan pengujian baru diperkuat dengan sumber lain.

## Bentuk masukan lokal bila 67 PASS

Notebook 68 menulis seluruh **230 pasien development**. Untuk tiap pasien:

- `candidate_scores.csv`: seluruh pusat kandidat dan tiga score.
- `candidate_crops.npz`: crop raw asli 12 lead, bentuk `[2,4,12,1600]`; tipe pertama kandidat PVC, kedua kandidat normal-N.
- `valid_beat_mask`: slot yang benar-benar mempunyai kandidat; padding nol selalu dimask.
- `centers`: indeks raw asli untuk setiap crop; slot kosong −1.
- `fallback_window`: window tengah tanpa tipe dan jumlah sampel valid, terpisah dari slot kandidat.
- `selection.json`: keputusan/QA/provenance, dengan `pvc_verified=False`, `sinus_verified=False` dan tidak menggunakan label anatomi untuk seleksi.

Pilih maksimal empat kandidat per tipe berdasarkan skor dengan tie-break indeks lalu urutkan waktu. Pasien tanpa salah satu/kedua tipe tetap disimpan; tidak memaksa beat menjadi PVC atau sinus. Crop disimpan dalam satuan native yang belum diverifikasi sebagai mV. Tidak ada otomatisasi training origin dari file ini: fase berikut perlu merancang kontrol all-candidate/fallback/mask dan ablation input yang sama bagi CNN versus CNN–Transformer.

## Resume dan penelusuran masalah

Source/runtime/protokol/runner difingerprint. File dataset, fitur, pipeline, sample training, prediksi dan manifest diverifikasi saat dibaca/resume. Proses mempunyai lock V12 sendiri; tidak menginisialisasi atau mengubah run V3–V11. Mengulang sel melewati shard/fit yang sudah lengkap dan sesuai hash. Stop kernel dapat mengulang fit yang belum selesai; fit L2 tidak mempunyai checkpoint per iterasi.

Log lengkap ada di `manual_logs/`, ditampilkan langsung di notebook. Saat error, kirim bagian log sebelum traceback; runner juga menampilkan 40 baris terakhir. Jangan mengubah source/protocol saat run sudah disiapkan; perbaikan source yang perlu merubah identitas menggunakan run baru atau recovery yang terpisah.

## Pekerjaan setelah V12

1. Analisis hasil 66/67 dan coverage 68 bila PASS. Jangan menyimpulkan identitas beat lokal telah tervalidasi hanya karena benchmark publik lulus.
2. Bila masukan layak, implementasikan ablation CNN compact → lead/rhythm-token Transformer pada tiga asal anatomis, dengan split/inner selection/patient loss yang sama dan masked fallback. Rancangan lengkap ada pada [kajian literatur](KAJIAN_LITERATUR_CNN_TRANSFORMER_PVC_20261007.md).
3. Fitur morfologi tanpa RR menjadi kontrol sementara berdasarkan V11; fusion encoder V11 tetap eksperimen terpisah yang menunggu cache asli.
4. Evaluasi asal pada pasien baru/eksternal dan penulisan kontribusi/keterbatasan tetap diperlukan. V12 tidak menghasilkan model origin final, skor 0,9 atau jaminan penerimaan Q1.
