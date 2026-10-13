# Menjalankan V10: transfer ECG-FM

**Notebook 51–55 kini selesai dan seluruh 130 kasus lolos audit. Tidak perlu menjalankannya ulang.** Mean macro-F1 random frozen **0,4415**, pretrained frozen **0,3744**; tail fold1/seed42 **0,3928**. [Analisis hasil](ANALISIS_HASIL_V10_20261006.md). **Lanjut notebook 56** untuk diagnosis tanpa training ulang, sesuai [panduan diagnosis](PANDUAN_DIAGNOSIS_V10_20261006.md).

Alur di bawah dipertahankan sebagai rujukan eksperimen V10 yang sudah selesai. Training dan ekstraksi ECG asli V10 telah dijalankan manual oleh pengguna; audit tambahan Codex membaca hasil/cache/checkpoint tanpa inference model baru. Kode dan notebook 51–55 dibekukan.

**Sediakan setidaknya 5 GiB ruang kosong sebelum notebook 52.** Saat persiapan tersedia sekitar 3,7 GiB. Program menjaga cadangan 3 GiB dan memperhitungkan artefak tahap berikutnya. Jangan menghapus hasil eksperimen lama. Tutup aplikasi berat sebelum memuat encoder 90,9 juta parameter. Angka 5 GiB adalah saran ruang kerja untuk fase ini, bukan jaminan pemakaian aplikasi lain.

| Notebook | Tindakan manual | Hasil yang diperiksa |
|---|---|---|
| [51: kompatibilitas dan persiapan](../../notebooks/51_V10_ECGFM_Kompatibilitas_dan_Persiapan.ipynb) | Cache bobot → kompatibilitas CPU → persiapan → smoke sintetis MPS | `Synthetic mps forward/backward PASS`; status nol fit |
| [52: ekstraksi development](../../notebooks/52_V10_Ekstraksi_Representasi_Development.ipynb) | Inference frozen encoder acak, lalu pretrained; seluruh window pasien | Cache lengkap 230 pasien; classifier belum dilatih |
| [53: pilot](../../notebooks/53_V10_Pilot_Probe_dan_Partial_FineTuning.ipynb) | Fold 1, seed 42, tiga arm masing-masing 9 inner + 1 outer | 30 kasus lengkap; ringkasan/heatmap inner; tail tetap pilot |
| [54: benchmark frozen probe](../../notebooks/54_V10_Frozen_Probe_CV_5Fold_3Seed.ipynb) | Random probe dan pretrained probe dipisah; lima fold/tiga seed head | Total 90 inner + 30 outer probe, termasuk kasus pilot |
| [55: analisis](../../notebooks/55_V10_Analisis_dan_Visualisasi_Transfer.ipynb) | Verifikasi prediksi/history/seleksi dan rekonstruksi metrik | Mean fold-F1, pooled OOF, recall kelas, grafik grid dan audit |

Seluruh **130 kasus** kini lengkap: 120 probe + 10 tail pilot, bukan 130 model final. Notebook 54 menguji **frozen probe** saja; fine-tuning tail tetap pilot satu fold. Full probe benchmark sudah `True`; arah berikut adalah diagnosis notebook 56, bukan mengulang pilot/grid ini.

## Apa yang dibandingkan

| Arm | Bagian yang dilatih | Bagian yang dibekukan |
|---|---|---|
| `random_probe` | Head linear tiga kelas | Seluruh encoder dengan inisialisasi acak, seed 424242 |
| `pretrained_probe` | Head linear tiga kelas | Seluruh encoder dari checkpoint pretraining resmi |
| `pretrained_tail` | Layer Transformer terakhir + head | CNN, positional encoding, dan 11 layer Transformer sebelumnya |

Random probe **bukan** encoder from-scratch yang dilatih penuh. Pembanding ini menguji kualitas representasi pretrained dibanding arsitektur acak yang sama pada probe linear. Baseline CNN–Transformer V8 tetap menjadi pembanding historis dengan preprocessing/pooling/kapasitas berbeda. Benchmark from-scratch penuh dan fine-tuning lintas fold merupakan pekerjaan lanjutan setelah bukti pilot.

## Grid dan pemilihan model

Head linear: learning rate **[1e-4, 1e-3, 1e-2]**. Tail pretrained: **[1e-6, 3e-6, 1e-5]**. Keduanya dikombinasikan dengan batch **[8, 16, 32] pasien**, menghasilkan sembilan skenario per arm/fold. Ini rentang kandidat, belum nilai terbaik. Head tail memakai LR head terpilih dari inner pretrained probe fold 1.

Head maksimum 100 epoch/patience 15; tail maksimum 30/patience 7. AdamW, weight decay 1e-4, cosine scheduler dan gradient clip 1. Class weight berasal dari frekuensi **pasien training**. Setiap pasien tampil satu kali per epoch; semua window pasien dipooling. Gradient accumulation mempertahankan denominator weighted cross-entropy per batch pasien untuk menghindari hilangnya efek class weight saat forward satu pasien.

Split tetap mengikuti development V8/V9. Pada fold 1: 147 pasien inner training, 37 inner validation, 46 outer validation. Skenario dan epoch dipilih hanya dari inner macro-F1; bila skenario seri, LR lebih kecil lalu batch lebih kecil. Epoch terbaik memakai maksimum inner pertama. Model outer dibuat baru, scaler hanya dari seluruh pasien outer training, lalu dilatih dengan jumlah epoch terpilih tanpa callback evaluasi outer. Setelah itu outer dinilai sekali. Seed refit probe 42/43/44; tail hanya seed 42/fold 1. **Test 59 pasien dan endpoint final tidak diaktifkan.**

## Input dan bobot

Input native 12 lead `[I, II, III, aVR, aVL, aVF, V1–V6]`, 5000 sampel pada 2000 Hz, diadaptasi menjadi 1250 sampel pada 500 Hz: tetap sekitar **2,5 detik**. Interpolasi linear dan rumus z-score mengikuti implementasi normalizer penulis, tetapi normalisasi dilakukan per window native. Model pretrained memakai segmen 5 detik; penggunaan window 2,5 detik adalah adaptasi project, belum bukti kecocokan distribusi. Tidak menyambung window overlap menjadi rekaman baru. Z-score per lead menghilangkan skala amplitudo antarleads, sehingga manfaat/kerugiannya untuk lokalisasi perlu diuji kemudian.

Checkpoint resmi memakai 12 layer, dimensi 768, FFN 3072 dan 12 attention head. YAML publik berbeda (24/1024/4096/16); program memakai konfigurasi **di dalam checkpoint** dan strict loading seluruh encoder. Detail bukti berada di [audit implementasi V10](IMPLEMENTASI_DAN_VALIDASI_V10_20261006.md).

Bobot dan cache besar berada di:

```text
/Users/daniandraaa/Library/Caches/TA-GUWEH-ECGFM/
  584219ea492cdeef2e19ffbdf9c6ecc874ba427e/
    ecg_fm_model_only.pt
    model_only.json
    experiments/20261006_ecg_fm_transfer_pilot_v10/features/
```

Cache lokal menghindari eviction iCloud Desktop yang sempat menyebabkan timeout. Model-only menyimpan konfigurasi dan semua tensor model persis; optimizer pretraining upstream tidak diperlukan. Ukuran bobot sekitar 347 MiB, token cache float32 sekitar 0,51 GiB. Checkpoint tail terpilih disimpan untuk sembilan skenario inner dan satu refit (~270 MiB total); optimizer hanya untuk resume aktif, dihapus setelah prediksi/manifest selesai. Jangan membersihkan cache ini saat fase belum selesai. Jika cache dihapus sistem, bobot dapat diunduh kembali; ekstraksi fitur perlu dijalankan lagi.

Hasil kecil/checkpoint terpilih/log berada di `results/research_v10/20261006_ecg_fm_transfer_pilot_v10/`. Log notebook menampilkan seluruh stderr, termasuk penyebab endpoint gagal. Kegagalan ruang kosong akan menyebut kebutuhan tahapnya; bebaskan ruang di luar hasil riset lalu ulang sel. Jika terhenti, ulang sel di **device/run-id yang sama**: ekstraksi resume per pasien, training resume dari epoch terakhir lengkap. Kasus selesai dilewati setelah verifikasi hash. Jangan mengubah source/protokol/runner selama melanjutkan run.

## Batas ilmiah yang harus ikut laporan

Pretraining ECG-FM mencakup **Ningbo**. Dataset lokalisasi Anda juga berasal dari **Ningbo First Hospital**; belum ada crosswalk identitas yang membuktikan ada/tidaknya pasien yang sama. Ini **potensi overlap yang belum terverifikasi**, bukan bukti leakage. Pilot V10 diberi penanda eksploratif dan tidak boleh diklaim sebagai evaluasi transfer pada pasien yang pasti belum pernah dilihat saat pretraining. [Paper ECG-FM](https://arxiv.org/html/2408.05178v2), [paper lokalisasi](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2022.809027/full), [sumber PhysioNet 2021](https://physionet.org/content/challenge-2021/1.0.3/).

Setelah pilot, prioritas publikasi adalah membuktikan independensi sumber atau memilih checkpoint dengan sumber bebas overlap yang dapat diaudit, menguji fine-tuning dan kontrol from-scratch yang setara, lalu evaluasi pasien independen. Historical test pernah diakses pada versi lama; tidak dapat menjadi confirmatory unseen test baru. Mean 15 skor fold/seed bukan 15 observasi independen. Tanpa ahli ECG, label anatomi sumber dipertahankan; attention/detektor tidak menjadi anotasi PVC atau validasi klinis. Skor 0,9 dan kelayakan Q1 tetap harus dibuktikan lewat hasil serta rancangan evaluasi.
