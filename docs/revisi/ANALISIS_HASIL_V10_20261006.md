# Analisis hasil V10 dan prioritas perbaikan

**Notebook 51–55 selesai dan seluruh 130 kasus lolos audit. ECG-FM pada recipe V10 belum meningkatkan lokalisasi tiga kelas.** Mean macro-F1 pretrained beku **0,3744**, dibanding encoder acak beku **0,4415**. Pilot fine-tuning layer terakhir juga turun pada pasien/seed yang sama. Langkah berikut adalah [notebook 56 untuk diagnosis tanpa training ulang](../../notebooks/56_Diagnosis_Transfer_V10_Tanpa_Training_Ulang.ipynb), sebelum memperluas fine-tuning.

## Hasil yang sudah selesai

| Pekerjaan | Bukti | Status |
|---|---|---|
| Notebook 51 | 6/6 sel kode; load/cache/kompatibilitas/persiapan/smoke MPS | Selesai, tanpa error tersimpan |
| Notebook 52 | 3/3 sel; 230 pasien, 2.274 window, kedua encoder dan token cache | Selesai, cache hash/shape/finite terverifikasi |
| Notebook 53 | 5/5 sel; pilot tiga arm fold1/seed42 | Selesai, 27 selector inner + 3 refit outer |
| Notebook 54 | 5/5 sel; kedua frozen probe lima fold/tiga seed head | Selesai, total 90 selector + 30 refit probe termasuk pilot |
| Notebook 55 | 6/6 sel; analisis seluruh hasil | Selesai, 99 inner + 31 outer |
| Audit tambahan | 130 checkpoint finite, 130 scaler training, 11 pilihan inner, 10 pembaruan bobot tail | Lolos |
| Anggaran aktual | 2.945 epoch dan 33.832 optimizer update | Sesuai history/scheduler/exposure pasien |
| Preservation | 3.045 file source, notebook dan hasil V2–V10 | Tidak berubah |

Seluruh prediksi, confusion matrix, macro-F1 dan lima-fold OOF direkonstruksi dari artefak tersimpan. Masing-masing dari enam set OOF memiliki tepat 230 pasien unik. Exposure setiap pasien training satu kali per epoch; class weight dan scaler hanya berasal dari training partisi. Epoch/skenario dipilih di inner validation dan outer refit fixed-last tidak mempunyai callback penilaian outer. Audit ini tidak melakukan training atau inference model ECG baru.

Bukti mesin dan tabel tersedia di [audit lengkap](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/analysis_20261006_complete/audit.json) dan [summary terverifikasi](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/analysis_20261006_complete/summary_verified.csv). Prediksi/hasil notebook asli dipertahankan; laporan tambahan berada di folder analisis terpisah.

## Perbandingan frozen probe

Angka berikut adalah mean per fold/seed untuk 15 evaluasi outer. SD menggambarkan variasi skor; 15 evaluasi menggunakan pasien/split/backbone berulang, sehingga bukan 15 observasi independen.

| Arm | Mean macro-F1 | SD | Recall LVOT | Recall RVOT cusp | Recall RVOT other |
|---|---:|---:|---:|---:|---:|
| Encoder acak beku + head | 0,4415 | 0,0682 | 0,6434 | 0,3710 | 0,3524 |
| Encoder pretrained beku + head | 0,3744 | 0,0539 | 0,5460 | 0,3494 | 0,2952 |

Pretrained turun **6,71 poin persentase macro-F1** dan lebih tinggi pada 4/15 pasangan fold/seed. Rata-rata per seed juga turun pada ketiganya: seed 42 **0,4460 → 0,3856**, seed 43 **0,4296 → 0,3785**, seed 44 **0,4489 → 0,3591**. Penurunan paling besar muncul pada fold 4–5; fold 1–2 mempunyai beberapa kenaikan kecil. Tidak ada manfaat pretrained yang konsisten pada recipe ini.

Mean pooled OOF berbeda dari mean fold karena F1 nonlinear: encoder acak **0,4434**, pretrained **0,3807**, selisih **−0,0627**. Bootstrap berpasangan dan berstrata kelas pada 230 pasien, dengan tiga seed head tetap bersama, menghasilkan interval persentil 95% **[−0,1358; +0,0161]** dari 2.000 replikasi. Interval melintasi nol; data ini tidak memastikan pretrained selalu lebih buruk. Bootstrap hanya mengukur ketidakpastian prediksi tersimpan, belum variasi refitting, pretraining seed atau pemilihan metode. [Rincian bootstrap](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/analysis_20261006_complete/paired_patient_bootstrap.json).

![Perbandingan probe dan pilot tail](../../results/research_v10/20261006_ecg_fm_transfer_pilot_v10/analysis_20261006_complete/verified_probe_and_tail_pilot.png)

## Pilot fine tuning layer terakhir

Perbandingan ini memakai **46 pasien outer fold1, seed 42** yang sama, bukan mean lima fold.

| Arm | Inner F1 terpilih | Outer F1 | Epoch refit | LR utama | Batch pasien |
|---|---:|---:|---:|---:|---:|
| Random frozen probe | 0,5850 | 0,4247 | 8 | 1e-2 | 8 |
| Pretrained frozen probe | 0,4864 | 0,4454 | 8 | 1e-3 | 16 |
| Pretrained tail + head | 0,5371 | 0,3928 | 6 | Tail 1e-6, head 1e-3 | 16 |

Tail meningkatkan inner F1 tetapi outer turun **5,26 poin persentase** terhadap pretrained frozen pada kasus yang sama. Recall tail: LVOT **0,5833**, cusp **0,3500**, other **0,2857**. Sepuluh checkpoint tail terbukti mempunyai perubahan parameter; penurunan bukan karena layer terakhir sama sekali tidak terlatih.

Pada selector tail terbaik, akurasi training online mencapai **0,7551** pada epoch 6, kemudian **0,8571** pada epoch 13; inner F1 terbaik tetap pada epoch 6. Refit outer mempunyai akurasi training online **0,7228**, tetapi F1 outer **0,3928**. Ini memotivasi pemeriksaan generalisasi/regularisasi. Angka akurasi tersebut dihitung selama update epoch, bukan evaluasi training dari checkpoint akhir; diagnosis 56 akan menghitung metrik training–validasi secara sebanding sebelum menyimpulkan overfitting.

## Yang dapat dan belum dapat disimpulkan

Tidak ditemukan mismatch hash, pasien/label, scaler, shape/finite, pemilihan epoch atau budget training dalam artefak yang diperiksa. Kedua matriks fitur memiliki 230 baris unik, centered rank 229 dan dimensi variabel; tidak ada collapse numerik sederhana berupa semua pasien menerima vektor yang sama. CPU forward sintetis pernah cocok dengan fungsi penulis, dan smoke MPS lolos. **Reproduksi CPU–MPS pada ECG asli serta konsistensi encoder–cache belum diuji**, sehingga diagnosis 56 tetap diperlukan.

Beberapa perubahan sekaligus membedakan V10 dari CNN–Transformer sebelumnya: input 500 Hz, z-score tiap lead/window, mean seluruh token/window pasien, dan encoder besar yang dibekukan. Mean Encoder V9 **0,5311** lebih tinggi dari kedua frozen probe V10, tetapi ini pembanding historis; perbedaan tersebut tidak mengisolasi efek pretraining. Random frozen probe bukan encoder from-scratch yang dilatih penuh.

Hipotesis input yang perlu diuji setelah diagnosis: pretraining memakai segmen **5 detik**, sedangkan V10 menggunakan **2,5 detik**; z-score tiap lead menghilangkan perbedaan skala amplitudo antarleads; mean pooling bisa mengurangi informasi bagian lokal. Ketiganya merupakan **hipotesis project**, belum penyebab penurunan yang terbukti. Paper ECG-FM meneliti diagnosis ECG/LVEF, bukan tiga kelompok lokalisasi anatomi ini. [Paper penulis ECG-FM](https://arxiv.org/html/2408.05178v2).

Sumber pretraining memuat Ningbo, dan kohort lokalisasi juga berasal dari Ningbo First Hospital. Belum ada crosswalk yang mengesahkan independensi pasien; kesamaan rumah sakit bukan bukti bahwa pasien sama. Historical test juga telah diakses pada fase lama. Keduanya membatasi klaim transfer/generalisation yang dapat dipakai untuk publikasi. [Sumber pretraining ECG-FM](https://arxiv.org/html/2408.05178v2), [paper lokalisasi](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2022.809027/full).

## Prioritas perbaikan berikutnya

1. **Jalankan notebook 56.** Reproduksi probabilitas enam checkpoint terpilih, ukur macro-F1 training–validasi, dan bandingkan encoding CPU/MPS/cache pada tiga pasien inner training yang dipilih deterministik per kelas. Tidak ada training ulang/test atau pemilihan model baru.
2. **Jika cache/backend konsisten, periksa kapasitas head dan regularisasi.** Ada 768 fitur untuk sekitar 146–148 pasien inner training; head current memakai weight decay 1e-4. Uji head linear dengan L2 yang dipilih pada inner pasien yang sama, agar manfaat representasi tidak hanya dinilai lewat satu recipe AdamW/early stopping. Logistic regression menyediakan regularisasi eksplisit; grid dan biaya harus ditetapkan sebelum fit. Ini langkah bersyarat setelah diagnosis, belum implementasi/hasil V11. [Dokumentasi resmi scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
3. **Ablation input/pooling satu perubahan per eksperimen.** Bandingkan normalisasi dan durasi kontinu yang sah tanpa menyambung window overlap secara semu. Gunakan kontrol random/pretrained dengan adapter yang sama. Pertahankan semua pasien dan nama kelas; skor tidak menjadi alasan menghapus kasus.
4. **Fine-tuning lebih dalam setelah baseline representasi jelas.** Sertakan kontrol arsitektur/input dari awal yang dilatih setara; pilot dulu dengan inner selection. Bukti V10 belum mendukung memperluas tail recipe sekarang ke lima fold.
5. **Perkuat klaim publikasi.** Audit independensi pretraining atau gunakan checkpoint bersumber independen yang dapat diverifikasi, lalu evaluasi pasien eksternal berlabel anatomi. Tanpa ahli ECG, attention/detektor tetap analisis algoritmik, belum verifikasi PVC/interpretasi klinis.

**Tidak perlu mengulang notebook 51–55.** Hasil saat ini bukan batas maksimum yang terbukti; ia menunjukkan transfer dengan recipe ini belum berhasil. Nilai 0,9 dan kecukupan Q1 memerlukan bukti pada evaluasi pasien yang sah, bukan kenaikan skor pada dataset yang sudah berulang kali dipakai untuk memilih metode.
