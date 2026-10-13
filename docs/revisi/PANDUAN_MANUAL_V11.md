# Panduan manual V11 — morfologi, RR, dan head berregularisasi

**Status terbaru — 7 Oktober 2026:** notebook 58 LUDB PASS, 59 selesai 230 pasien, dan **60/61 kini selesai masing-masing 11/11 sel tanpa error untuk tiga arm fitur**. Sebanyak **150 fit termasuk pilot** selesai: 135 inner + 15 outer, dengan pilihan C dan tiga OOF penuh terverifikasi. Mean fold F1 **QRS/RR 0,4210; morfologi 0,5555; morfologi + RR 0,5532**. **Tidak perlu menjalankan ulang tiga arm fitur atau notebook 57–59.** Empat arm encoder/fusion masih `TERTUNDA`; anggaran 350 di bawah adalah rencana penuh, bukan 350 fit selesai. [Hasil, grafik dan tindakan berikutnya](ANALISIS_HASIL_FITUR_V11_20261007.md).

Langkah manual berikutnya: pulihkan cache asli yang cocok dengan audit V10, lalu lanjutkan **hanya sel empat arm yang tertunda pada 60/61** sesudah sel cache menunjukkan ready. Jika tidak ada backup, diperlukan endpoint regenerasi representasi yang dipisahkan dari hasil lama; jangan langsung mengulang notebook 52 atau memakai cache berbeda dalam run yang sama. Pilihan C masing-masing fold belum satu hyperparameter model final. Tidak ada training CNN–Transformer atau final/test baru pada audit fitur ini.

## Riwayat persiapan dan perbaikan sebelum hasil lengkap

**Perbaikan import notebook 60/61 — 7 Oktober 2026:** error `ModuleNotFoundError: research_v11_recovery` berasal dari sel setup yang hanya memasukkan folder `notebooks` ke `sys.path`. Kedua notebook sekarang juga memasukkan root project. **Restart Kernel, pilih `.venv`, lalu jalankan notebook 60 dari sel pertama.** Run tetap `_r1`; source/protokol/backend tidak berubah, sehingga hasil 58–59 tetap digunakan. Sel setup/cache berhasil diuji dari cwd `notebooks/` dengan interpreter terisolasi, tanpa training. Notebook 58 sudah menunjukkan LUDB PASS (F1 deteksi 0,96853; bukan F1 classifier lokalisasi), dan 59 telah mengekstrak seluruh 230 pasien. Cache encoder masih belum tersedia: pesan TERTUNDA pada empat arm encoder/fusion adalah status prasyarat; tiga arm morfologi dapat dilanjutkan setelah gate yang sudah lulus. Byte copy 60/61 sebelum perbaikan tersimpan di `arsip/notebooks_sebelum_perbaikan_import_v11_20261007/`.

**Pembaruan 7 Oktober 2026:** error notebook 58 berasal dari guard disk, kemudian cache V10 yang tidak tersedia; parser juga diperbaiki untuk anotasi LUDB yang tidak lengkap. Notebook 57–61 kini memakai runner perbaikan dan run **`20261007_morphology_regularized_features_v11_r1`**, dengan source/hasil awal dipertahankan. **Restart Kernel → 57 dari sel pertama → 58 dari sel pertama**, gunakan `DOWNLOAD_LUDB=False` karena file sudah diunduh. [Diagnosis dan langkah lengkap](PERBAIKAN_NOTEBOOK_58_CACHE_DAN_ANOTASI_20261007.md). Tiga arm fitur dapat berjalan setelah LUDB PASS tanpa embedding; empat arm encoder/fusion tetap tertunda sampai cache terkait tersedia dan cocok dengan audit V10. Anggaran 350 di bawah adalah seluruh rencana, bukan jumlah fit yang sudah tersedia.

**Status 6 Oktober 2026: implementasi tersedia; eksperimen ECG asli belum dijalankan.** Notebook 56 telah selesai. Fase ini menjawab usulan fitur QRS/RR/lead dan diagnosis gap training–validasi V10. Tidak perlu mengulang notebook 00–56. Mulai **57 → 58 → 59**, lalu pilot 60 dan CV 61 setelah pengukuran lulus.

Pilih kernel Python `.venv` project di VS Code/Jupyter. Jalankan sel satu per satu dengan **Shift+Enter**, agar log/progres terlihat. Runner selalu memakai interpreter `.venv/bin/python` dan CPU. WFDB yang sudah tersedia di `.ecgqa` dipakai untuk LUDB; tidak ada instalasi atau perubahan environment otomatis.

| Notebook | Tindakan saat kamu menjalankan | Hasil yang diperiksa |
|---|---|---|
| [57 — Persiapan](../../notebooks/57_V11_Persiapan_Fitur_dan_Regularisasi.ipynb) | Verifikasi protokol/fold/cache V10, lalu smoke sinyal dan classifier sintetis CPU | 230 pasien development, 5 fold, 7 arm, 350 fit terencana; smoke PASS |
| [58 — LUDB](../../notebooks/58_V11_Validasi_Pengukuran_QRS_LUDB.ipynb) | Download resmi ketika `DOWNLOAD_LUDB=True`, hash file, uji deteksi dan onset/offset 200 rekaman | `ludb/summary.json`, coverage dan galat per lead/rekaman; gate PASS/FAIL |
| [59 — Ekstraksi/QA](../../notebooks/59_V11_Ekstraksi_dan_QA_Fitur_Morfologi.ipynb) | Ekstraksi 230 rekaman kontinu native, audit missingness dan 3 pratinjau tetap | 132 fitur/pasien, kualitas, estimasi batas per kandidat, semua pasien dipertahankan |
| [60 — Pilot fold1](../../notebooks/60_V11_Pilot_Fitur_dan_Fusion_Fold1.ipynb) | Fit 9 C inner + 1 refit outer per arm | 70 fit termasuk dalam full CV; F1, recall, gap dan convergence |
| [61 — Lima fold](../../notebooks/61_V11_CV_5Fold_dan_Analisis_Fitur.ipynb) | Lengkapi semua arm/fold, verifikasi prediksi dan pemilihan C, analisis/plot | 350 fit total; 280 baru bila seluruh pilot selesai; perbandingan paired dan OOF |

**Catatan saat implementasi 6 Oktober 2026:** belum ada hasil peningkatan akurasi V11 pada saat itu. Hasil aktual kini tercantum pada status terbaru dan laporan di atas. Uji kode sintetis tidak menjadi hasil ECG asli atau validasi klinis. Tidak ada training CNN/Transformer baru pada fase fitur ini; empat arm yang memakai representasi V10 masih menunggu cache. Hasil V2–V10, source dan runner lama dipertahankan.

## Mengapa menguji head L2 terlebih dahulu

Notebook 56 memeriksa enam fit (tiga inner dan tiga outer). Pada **tiga refit outer fold1/seed42**, gap training–validasi sekitar **0,318–0,391 macro-F1**. Angka itu konsisten dengan masalah generalisasi pada kasus yang diperiksa, tetapi belum membuktikan satu penyebab tunggal. V11 memberi kontrol yang lebih sederhana sebelum menambah kapasitas model.

Grid baru adalah **C**, inverse regularization strength logistic regression:

`[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`.

C lebih kecil berarti regularisasi L2 lebih kuat. Solver LBFGS melakukan optimisasi deterministik; LR dan batch size tidak menjadi parameter head ini. Grid LR × batch pada eksperimen neural sebelumnya tetap menjadi riwayat penelitian. Fase V11 adalah ablation fitur/head, bukan klaim fine-tuning CNN baru.

| Arm | Input mentah per pasien | Pertanyaan |
|---|---:|---|
| `qrs_rr` | 32 | Apakah estimasi durasi QRS dan konteks RR memberi informasi? |
| `morphology` | 126 | Apakah morfologi antarlead membantu tanpa RR? |
| `morphology_rr` | 132 | Apakah menambah RR pada morfologi membantu atau merugikan? |
| `random_l2` | 768 | Bagaimana cache encoder random frozen dengan head L2? |
| `pretrained_l2` | 768 | Bagaimana cache ECG-FM pretrained frozen dengan head L2? |
| `fusion_random` | 900 | Apakah menggabungkan 132 fitur dengan cache random membantu? |
| `fusion_pretrained` | 900 | Apakah menggabungkan 132 fitur dengan cache pretrained membantu? |

Jumlah kolom setelah imputer dapat bertambah karena missing indicator yang ditentukan hanya dari training. Fusion berupa concatenation fitur untuk head L2, belum menjadi jaringan multimodal yang dilatih end-to-end. `random_l2` bukan CNN from-scratch terlatih: encoder random tetap frozen.

## Pengukuran dan batas interpretasinya

Ekstraksi membaca rekaman kontinu **2.000 Hz**, mempertahankan urutan `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]`. Semua kandidat konsensus detektor V3 digunakan tanpa maksimum window K4/K8. Kandidat lengkap memerlukan konteks 250 ms sebelum dan 300 ms sesudah; tetangga RR tetap diambil dari daftar kandidat seluruh rekaman sebelum pemilihan crop lengkap.

Cabang morfologi memakai filter 0,5–100 Hz; cabang delineation tersendiri 0,5–40 Hz dari raw, envelope gradien absolut, dan quiet-run yang mengapit aktivitas inti QRS. Ambang dan parameter dicatat pada `research_v11/protocol.yaml`. Ini heuristik rekayasa yang diuji, bukan reproduksi penuh metode klinis/paper atau delineator yang sudah terbukti.

Untuk setiap lead, fitur mencakup estimasi durasi dalam ms, amplitudo R/S relatif, R/(R+S), dan integral bertanda QRS dibagi RMS bersama (satuan relatif × detik). Baseline lokal diambil sebelum estimasi onset. Satu RMS per rekaman digunakan **bersama seluruh lead**, sehingga skala relatif antarlead tetap tersedia. Satuan raw belum dikonfirmasi sebagai mV; tidak ada threshold amplitudo klinis dalam mV.

Fitur tambahan: rentang QRS antarlead robust (persentil onset 10%/offset 90%, minimum 6 lead valid), nomor V pertama dengan R≥S, V2S/V3R bila denominator cukup, RR sebelum/sesudah dan rasionya. Rentang robust bukan definisi klinis earliest-onset/latest-offset. Indeks V pertama R-dominan bukan diagnosis zona transisi lengkap. Nomor lead fisik menentukan **kolom**, bukan scalar konstan yang diharapkan sendiri membedakan pasien.

Median/IQR dari seluruh kandidat menghasilkan 132 kolom. Batas yang tidak dapat diukur atau durasi di luar rentang rekayasa 35–240 ms menjadi NaN, **tidak menghapus pasien**. Ini bukan aturan diagnosis PVC. Kualitas dicatat terpisah dan tidak digunakan sebagai input head; imputer mengisi nilai hanya berdasarkan pasien training tiap fit.

**Kandidat QRS belum terverifikasi PVC.** Median seluruh kandidat bisa mencampur sinus dan PVC sehingga mengurangi informasi lokalisasi. RR sebelum kandidat belum otomatis coupling interval PVC, dan RR setelahnya belum otomatis compensatory pause. Label asal anatomi per pasien yang tervalidasi tidak memvalidasi identitas setiap denyut.

## Gate LUDB

[LUDB 1.0.1](https://physionet.org/content/ludb/1.0.1/) berisi 200 rekaman 10 detik/12 lead pada 500 Hz dengan anotasi batas gelombang oleh kardiolog. Dipakai hanya untuk menguji pengukuran, tanpa menambahkan pasien LUDB ke classifier lokalisasi. Download berkas resmi dilakukan saat kamu menjalankan notebook 58; checksum resmi diperiksa dan resume tidak menimpa berkas berbeda secara diam-diam.

Gate rekayasa yang telah ditetapkan sebelum run:

- Micro detection F1 ≥0,95 pada region anotasi masing-masing lead, matching satu-ke-satu ±100 ms.
- Coverage batas ≥0,90 pada seluruh QRS anotasi, termasuk QRS yang tidak terdeteksi.
- Median galat absolut onset ≤20 ms, offset ≤20 ms, dan durasi ≤25 ms.
- Coverage/galat harus lulus pada **anchor anotasi dan anchor hasil deteksi**; hasil oracle saja tidak cukup.

P95 galat juga dilaporkan. Beat yang sama berulang di 12 lead tidak dianggap pasien independen. PASS belum menjamin performa delineation pada semua ECG lokal 2.000 Hz, apalagi memastikan beat PVC atau validasi klinis lokasi asal.

Jika FAIL: simpan `summary.json`/`boundary_errors.csv`; jangan mengubah ambang untuk menyatakan lulus. Notebook 59 tetap dapat dipakai untuk diagnosis ekstraksi. Dua baseline `random_l2`/`pretrained_l2` dapat dijalankan terpisah, tetapi arm fitur/fusion diblokir. Perbaikan algoritme perlu versi/run baru; jika LUDB telah dipakai untuk memilih perbaikan, klaim generalisasi pengukuran memerlukan data anotasi lain yang belum digunakan untuk tuning.

## Split, seleksi, dan analisis

Seluruh fold pasien diwarisi dari V10. Masing-masing arm/fold memilih C pada **inner training/validation** yang sudah tersimpan. Bila skor seri, C terkecil dipilih. Setelah itu dibuat pipeline baru untuk seluruh outer training. Outer validation tidak dipakai untuk imputer/scaler/bobot kelas, early stopping atau pemilihan C. Tidak mengulang seed identik untuk menghasilkan replikasi semu dari classifier deterministik.

Target utama laporan: **mean macro-F1 pasien lima outer fold**. Pooled OOF adalah agregasi terpisah untuk 230 pasien yang diprediksi sekali per arm. Dibandingkan secara paired pada fold/pasien sama, terutama fusion versus kontrol L2 dari encoder yang sama. Perhatikan recall RVOT cusp dan septal/freewall/other, konsistensi fold, galat pengukuran, missingness, convergence, serta gap training–validasi.

Bootstrap 2.000 kali stratified per kelas menggunakan prediksi OOF tetap; intervalnya **conditional pada prediksi tersebut** dan tidak mencakup ketidakpastian pemilihan metode/training ulang. Audit memverifikasi manifest, identitas split, metrik train/evaluation semua fit lengkap, dan pilihan sembilan C yang mendasari tiap outer refit. Analisis tidak memuat ulang classifier untuk inferensi baru.

V11 tetap eksploratif karena metode telah berkembang berdasarkan hasil development V3–V10. Historical test yang sudah pernah dilihat tidak menjadi unseen test baru. Belum ada endpoint final/test; **0,9 dan publikasi Q1 belum dapat dijanjikan**.

## Penyimpanan dan melanjutkan error

Run aktif: `results/research_v11/20261007_morphology_regularized_features_v11_r1/`; run awal `20261006_morphology_regularized_features_v11/` dipertahankan sebagai riwayat. Dataset anotasi: `external_data/ludb_1.0.1/`. Log tiap endpoint ada di `manual_logs/`. Hasil patient/record/fit lengkap akan diverifikasi lalu dilewati ketika sel diulang. Fit yang terputus dimulai ulang, bukan melanjutkan epoch; head ini memakai solver convex.

Cadangan ruang minimum 3 GiB dijaga sebelum tahap yang menulis. Ruang saat persiapan terbatas; jika guard storage berhenti, kosongkan file non-riset yang kamu pilih dan ulang sel. Program tidak menghapus checkpoint/data lama. Jika solver gagal converge, ia menyimpan kegagalan dan tidak menerbitkan skor untuk seleksi.

Rencana penelitian dan prioritas lanjutan: [riset dan fase V11](RISET_DAN_FASE_FITUR_MORFOLOGI_V11_20261006.md).

## Riwayat pemeriksaan implementasi pada 6 Oktober 2026

**12 uji sintetis CPU lulus**: batas QRS biphasic, sampling 500/2.000 Hz, polaritas, invariansi skala bersama, RR kontinu, kegagalan tetap NaN, parsing anotasi, denominator beat yang terlewat, gate oracle versus operational, preprocessing training-only, pemilihan C/resume dan audit prediksi/manifest. Uji ini tidak membaca ECG lokal atau mengunduh LUDB. Kelima notebook valid, semua sel kode belum dieksekusi dan tanpa output tersimpan.

Endpoint `prepare` dan smoke sintetis juga sudah diperiksa; persiapan mengikat 230 pasien/fold/cache V10 tanpa encoder inference atau training ECG asli. Notebook 57 tetap perlu kamu jalankan untuk melihat log sendiri. Snapshot **873 file lama** (source/runner/notebook 00–56 dan JSON V10 yang tercantum dalam snapshot) tidak berubah; ini bukan audit checkpoint lama secara penuh yang baru. Belum ada download/benchmark LUDB, ekstraksi fitur lokal, fit ECG asli, atau skor V11.
