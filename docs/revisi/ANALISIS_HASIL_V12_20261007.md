# Analisis hasil V12 — identitas beat, 7 Oktober 2026

**Notebook 62–67 selesai, 40/40 fit CV dan satu deployment terverifikasi. Evaluasi MIT-BIH gagal memenuhi gate; notebook 68 melewati ekspor sebagaimana dirancang.** Tidak perlu mengulang notebook ini dengan konfigurasi yang sama. Prioritas berikut ialah perbaikan detector/pusat beat, kemudian generalisasi classifier dan kalibrasi score.

V12 menguji **jenis beat N/V-r/other**, bukan tiga asal anatomis LVOT/RVOT cusp/RVOT septal-free-wall-other. Angka V12 tidak menggantikan macro-F1 pasien pada V11 atau hasil historical origin test. Kandidat lokal belum diekspor, sehingga belum ada masukan pasangan PVC–normal-N atau model lokalisasi baru dari fase ini.

## Status masing-masing notebook

| Notebook | Sel kode tersimpan | Pekerjaan dan hasil |
|---|---|---|
| 62 | 5/5, tanpa error | Persiapan 75 INCART/32 grup, lima outer fold; smoke sintetis lulus. Percobaan awal berhenti pada guard disk, tetapi percobaan berikutnya selesai |
| 63 | 4/4, tanpa error | 75/75 shard fitur; 18.735 dari 20.007 reference PVC eligible matched kandidat, recall detector **93,64%** |
| 64 | 5/5, tanpa error | Delapan fit pilot fold 1; F1 morfologi **0,7321**, morfologi+RR **0,8558** |
| 65 | 4/4, tanpa error | Melengkapi semua **40 fit**: 30 inner dan 10 outer; pilot digunakan kembali |
| 66 | 5/5, tanpa error | OOF/visualisasi lengkap; memilih morfologi+RR, C **0,01**, threshold PVC **0,5** dari INCART; satu fit deployment pada 32 grup/61.556 row lead |
| 67 | 5/5, tanpa error | Seluruh 48 MIT-BIH/47 grup dievaluasi; precision **0,3371**, recall **0,4912**, F1 **0,3998**, gate **FAIL** |
| 68 | 4/4, tanpa error | Membaca gate FAIL dan melewati ekspor; tidak ada `local_candidates/completed.json` |

Execution count menunjukkan sel telah dijalankan, bukan bukti tahap ekspor berhasil. Pada 68, output STOP/dilewati adalah hasil yang diharapkan untuk gate FAIL. Tidak ditemukan error runtime tersimpan pada ketujuh notebook versi terbaru. Log gagal persiapan pertama tetap dipertahankan sebagai riwayat.

## Hasil CV INCART

| Arm | Mean precision | Mean recall | Mean fold F1 ± SD | Pooled OOF F1 |
|---|---:|---:|---:|---:|
| Morfologi | 0,7156 | 0,6707 | **0,6793 ± 0,1338** | 0,6859 |
| Morfologi + RR | 0,7694 | 0,8204 | **0,7897 ± 0,1438** | 0,8041 |

Mean fold F1 dan pooled OOF F1 berbeda cara agregasinya; keduanya tidak boleh dipertukarkan. Pooled OOF menjumlahkan TP/FP/FN seluruh rekaman, dengan parameter yang dipilih secara inner pada tiap fold. Deployment adalah satu refit dari seluruh INCART, bukan salah satu checkpoint OOF.

RR meningkatkan mean fold F1 **11,04 poin persentase** dan meningkatkan F1 pada kelima fold. Namun ketidakstabilan masih besar:

| Fold | F1 morfologi | F1 morfologi + RR | Pilihan C/threshold RR |
|---|---:|---:|---|
| 1 | 0,7321 | 0,8558 | 0,01 / 0,5 |
| 2 | 0,6962 | 0,8720 | 0,01 / 0,5 |
| 3 | 0,8489 | 0,8605 | 0,01 / 0,5 |
| 4 | 0,4848 | 0,5342 | 0,01 / 0,7 |
| 5 | 0,6348 | 0,8262 | 0,01 / 0,5 |

Fold 4 tetap sulit untuk kedua arm. CI bootstrap grup pooled F1 RR **0,6825–0,8825**; interval bersyarat pada prediksi OOF tetap, bukan ketidakpastian seluruh retraining/seleksi. Hasil ini mendukung manfaat RR untuk identitas beat pada INCART. Pada V11 RR belum membantu **asal anatomis**; target dan dataset berbeda, sehingga temuan tersebut tidak bertentangan.

## Evaluasi lintas dataset: gate FAIL

| Ukuran MIT-BIH | Aktual | Ambang protokol |
|---|---:|---:|
| PVC precision | **0,3371** | ≥0,80 |
| PVC recall end-to-end | **0,4912** | ≥0,80 |
| PVC F1 end-to-end | **0,3998** | ≥0,80 |
| Precision normal-N ber-score ≥0,90 | **0,7801** | ≥0,80 |
| Normal-N reference yang terpilih benar | **6,63%** | ≥25% |
| Pilihan normal-N tidak kosong | Ya | Wajib |

Hanya syarat pilihan normal tidak kosong yang lulus. Reference normal-N berjumlah 75.019; 4.974 terpilih benar pada threshold 0,90. Precision mengabaikan Q/unclassifiable, dan terdapat sembilan prediksi normal serta empat prediksi PVC pada kandidat matched Q, yang dilaporkan terpisah. Threshold/gate tidak diubah setelah melihat hasil.

PVC: **TP 3.502, FP 6.887, FN 3.627**, dari **7.129 reference PVC eligible**. CI bootstrap 47 grup F1 **0,2651–0,5416**. Matched-known ROC-AUC PVC **0,9211** dan PR-AUC **0,6580** hanya mengukur kandidat matched yang labelnya diketahui; detector miss dan false alarm unmatched tidak masuk metrik conditional tersebut. AUC 0,92 bukan F1 pipeline 0,92.

![Ringkasan audit dan pembagian kesalahan V12](../../results/research_v12/20261007_beat_identity_v12/analysis_20261007_complete/v12_failure_overview.png)

## Diagnosis 1 — mayoritas false positive berasal dari kandidat tanpa reference cocok

| Sumber false positive | Jumlah | Proporsi seluruh FP |
|---|---:|---:|
| Kandidat tidak matched reference dalam toleransi 100 ms | **4.239** | **61,55%** |
| Beat N diprediksi PVC | 1.486 | 21,58% |
| Beat other diprediksi PVC | 1.162 | 16,87% |

Unmatched adalah status pencocokan evaluasi; belum diagnosis klinis tentang waveform. Median jarak unmatched-FP ke reference terdekat **311 ms**, p10–p90 **281–644 ms**. Pada rekaman 117 yang tidak memiliki reference V/r eligible, detector menghasilkan **3.007 kandidat untuk 1.534 reference**, dan 1.464 false alarm PVC berasal dari kandidat unmatched. Median jarak kasus tersebut sekitar **297 ms**. Rekaman 113 dan 123 menghasilkan 788 dan 767 unmatched-FP, dengan median jarak sekitar 331 dan 328 ms.

Ini bukti kuantitatif deteksi tambahan di luar anotasi beat. **Hipotesis** yang perlu diperiksa melalui benchmark/visualisasi ialah puncak non-QRS atau pusat beat keliru. Waveform mentah tidak dibaca pada audit ini, sehingga tidak dinyatakan bahwa setiap kandidat tambahan adalah T-wave.

RR dihitung dari urutan kandidat detector. Bila urutan mempunyai deteksi tambahan, interval RR juga berubah; hal ini dapat ikut menyesatkan classifier yang menganggap interval pendek sebagai ciri PVC. Hubungan ini merupakan dugaan mekanisme berdasarkan desain pipeline, belum ablation sebab-akibat yang terukur.

Training V12 hanya memuat kandidat matched berlabel diketahui; kandidat unmatched tidak diajarkan sebagai kelas background/rejection. Karena itu, classifier belum memperoleh supervisi eksplisit untuk menolak tipe kandidat tambahan tersebut.

## Diagnosis 2 — detector menghilangkan banyak PVC sebelum classifier

Detector hanya mencocokkan **4.813/7.129 PVC**, recall **67,51%**, dibanding **93,64%** pada INCART. Dari 3.627 FN total, **2.316 (63,85%)** terlewat sebelum classification dan 1.311 terjadi setelah deteksi.

| Rekaman | Reference PVC | PVC terlewat detector | Recall detector PVC |
|---|---:|---:|---:|
| 208 | 992 | **804** | 18,95% |
| 233 | 830 | **531** | 36,02% |
| 119 | 444 | **355** | 20,05% |
| 213 | 220 | 123 | 44,09% |
| 228 | 362 | 103 | 71,55% |

Tiga rekaman pertama menyumbang **72,97% seluruh detector miss PVC**. Dengan kandidat yang sama, classifier sempurna sekalipun hanya bisa mencapai recall 67,51%; perubahan threshold classifier saja tidak dapat memenuhi syarat recall 80%. Batas ini berlaku pada kandidat V12 saat ini. Perbaikan detektor dapat mengubahnya; angka tersebut bukan batas intrinsik dataset atau model lokalisasi.

Semua 2.316 detector miss PVC terdapat dalam 40 rekaman dengan MLII/V1 dan aturan konsensus dua kanal. Hal ini menjadikan konsensus/seleksi kanal kandidat utama untuk ablation. Belum dapat disimpulkan bahwa aturan konsensus adalah satu-satunya sebab karena detector per kanal belum dievaluasi terpisah.

## Diagnosis 3 — transfer kanal/jenis beat dan confidence belum stabil

Pada rekaman yang memakai MLII tunggal, F1 pooled **0,0507** dengan 2.407 FP; pada dua rekaman fallback V5, F1 **0,0485** dengan 195 FP; pada MLII/V1, F1 **0,4655**. Ini analisis tambahan yang mempertahankan semua rekaman, bukan alasan membuang konfigurasi kanal yang sulit. Kelompok V5 hanya mempunyai enam reference PVC, sehingga metriknya sangat tidak stabil.

INCART dan MIT-BIH berbeda distribusi label/kanal. Metadata anotasi MIT-BIH memuat banyak L/left bundle-branch, paced dan fusion paced; kategori tersebut tidak atau jauh lebih sedikit hadir di INCART. FP known-other di antaranya L **514**, fusion F **208**, A **185**, R **91**. Sampling dan kanal sebenarnya tetap dibaca dari metadata; MLII tidak dinamai ulang sebagai lead II standar.

Perbedaan sumber, komposisi beat, pemilihan kanal, kandidat dan score berbobot adalah kemungkinan penyumbang transfer gap. Audit ini tidak menetapkan besar efek masing-masing. Normal-N coverage 6,63% pada score 0,90 juga menunjukkan bahwa score tinggi belum dapat diandalkan untuk memperoleh pasangan lokal secara luas.

## Prioritas berikutnya

1. **Benchmark detector QRS lebih dahulu, pada run baru.** Bandingkan heuristik V12 dengan implementasi XQRS, lalu deteksi per kanal dan konsensus/fallback yang prespecified. Ukur recall PVC sekaligus FP seluruh beat, duplicate/extra events, galat pusat dan kelompok lead. [Dokumentasi primer WFDB XQRS](https://wfdb.readthedocs.io/en/latest/processing.html#wfdb.processing.XQRS) menjelaskan adaptive threshold, backsearch dan inspeksi T-wave yang dapat dikonfigurasi. XQRS adalah kandidat pembanding, bukan metode yang sudah terbukti lebih baik pada data ini; default inspeksi T-wave perlu diperhatikan.
2. **Bentuk kandidat dan RR kembali dari detector terpilih.** Pertahankan tol matching 100 ms sebagai pembanding awal. Memperlebar toleransi atau refractory hanya supaya metrik naik dapat menutupi deteksi keliru/kehilangan PVC dan tidak menyelesaikan masalah.
3. **Uji rejection/background serta generalisasi classifier.** Background harus ditetapkan dari training beranotasi dengan margin aman dan aturan ambigu/Q, bukan dipaksakan menjadi label normal atau dipilih dari error eksternal. Uji CNN beat compact melawan baseline fitur dengan kelompok pasien/sumber terpisah, lead dropout/kanal aktual dan supervisi beat yang sama. Perubahan desain/loss pada sumber yang telah dilihat harus dinyatakan.
4. **Pilih dan kalibrasikan score pada training/validation yang disisihkan berdasarkan grup.** Pastikan precision serta coverage normal-N dan end-to-end PVC memenuhi tujuan; score classifier berbobot saat ini belum calibrated probability. Tidak menurunkan gate V12 setelah FAIL.
5. **Perbarui peran dataset dengan jujur.** MIT-BIH kini telah diamati dan digunakan untuk diagnosis. Bila dijadikan bahan desain atau training selanjutnya, jangan disebut pristine unseen test lagi. Evaluasi generalisasi final memerlukan sumber baru yang independensinya diperiksa atau protokol leave-one-source-out yang dinyatakan sejak awal. Anotasi beat publik tetap tidak menyediakan label lokasi PVC eksternal yang sama dengan target lokal.
6. **Ablation CNN–Transformer asal anatomis menyusul setelah mutu kandidat didukung bukti.** Pertahankan kontrol all-candidate, masked fallback dan seluruh 230 pasien. Empat arm fusion V11 yang menunggu cache adalah pekerjaan terpisah; hasil V12 tidak mengharuskan pengulangan V11.

Prioritas 1–4 belum diimplementasikan sebagai eksperimen baru pada audit ini. Semua fitting/inference ECG berikutnya tetap dijalankan manual oleh pengguna. Tidak perlu mencoba memaksa notebook 68 pada V12 yang gate-nya FAIL.

## Bukti audit

Audit terpisah `analysis_v12.completed` tidak mengubah paket/protokol model V12. Pemeriksaan meliputi **40 fit/100 skor evaluasi**, identitas source/runtime, hash fitur/checkpoint, reproduksi metrik dari prediksi, sample training hanya matched-known, cap/bobot grup, pasangan sampel C/arm, cakupan OOF unik, 47 grup eksternal termasuk 201/202 serta gate/CI.

Audit tidak melakukan fit, load model, inference atau pembacaan waveform. Ia membaca fitur/prediksi/reference tersimpan dan menghitung hash aset sumber. Checkpoint diverifikasi melalui hash; tidak diklaim diuji load model ulang.

- [Audit independen](../../results/research_v12/20261007_beat_identity_v12/analysis_20261007_complete/audit.json).
- [Audit setiap fit](../../results/research_v12/20261007_beat_identity_v12/analysis_20261007_complete/case_audit.csv).
- [Rincian kesalahan per rekaman/kanal](../../results/research_v12/20261007_beat_identity_v12/analysis_20261007_complete/external_failure_by_record.csv).
- [Grafik ringkasan](../../results/research_v12/20261007_beat_identity_v12/analysis_20261007_complete/v12_failure_overview.png).
- [Protokol/panduan sebagai referensi run yang selesai](PANDUAN_MANUAL_V12.md).

**Preservasi setelah audit:** 1139 berkas dalam snapshot run V12/source/notebook/aset dataset tetap identik; tidak ada perubahan atau file hilang. [Baseline](../../tmp/v12_completed_before_audit_20261007.json) · [Hasil preservasi](../../tmp/v12_completed_preservation_after_audit_20261007.json). Laporan/grafik audit ditulis pada folder baru yang terpisah dari output asli.
