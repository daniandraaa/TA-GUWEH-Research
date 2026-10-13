# Implementasi dan validasi V10 — 6 Oktober 2026

**Fase transfer ECG-FM siap untuk eksekusi manual notebook 51–55.** Persiapan memakai metadata development yang sama dengan V9, tanpa membaca test atau menjalankan training/inference ECG asli. Panduan pelaksanaan: [PANDUAN_MANUAL_V10.md](PANDUAN_MANUAL_V10.md).

Run: `20261006_ecg_fm_transfer_pilot_v10`. Hash eksperimen:

```text
423cc526c4f317d55561bf05b702db30a829ff8296f013f0e58347d59b526f91
```

Package `research_v10/`, runner `_v10_runner.py`, protocol dan hasil V10 terpisah. `.venv` lama dipakai tanpa instalasi atau perubahan dependency. Source numerik official `fairseq_signals` dipin dan disimpan dengan lisensi; hanya jalur import internal yang diubah. Wrapper memakai forward inference tanpa masking/SSL head, dan fungsi forward penulis dieksekusi sebagai referensi pada modul numerik yang sama. Ini menguji susunan wrapper pada CPU; belum membuktikan seluruh backend fairseq/CUDA atau MPS mempunyai hasil bit-identik.

## Identitas bobot

Checkpoint **pretraining**, bukan classifier diagnosis MIMIC: `mimic_iv_ecg_physionet_pretrained.pt` dari [rilis resmi](https://huggingface.co/wanglab/ecg-fm/tree/584219ea492cdeef2e19ffbdf9c6ecc874ba427e). HF revision `584219ea492cdeef2e19ffbdf9c6ecc874ba427e`; source numerik revision `f8f0ff1c788a82c2059cb452cd5462898867489e` dari [fairseq-signals](https://github.com/Jwoo5/fairseq-signals/tree/f8f0ff1c788a82c2059cb452cd5462898867489e).

SHA256 upstream:

```text
4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e
```

Unduhan **1.090.825.421 byte** diverifikasi utuh sebelum parsing dengan `torch.load(weights_only=True)`. File memuat optimizer upstream; konfigurasi dan seluruh **215 tensor model** disalin ke file model-only di Library/Caches. Kesetaraan setiap tensor diuji dengan `torch.equal`, konfigurasi diuji sama, lalu checksum turunan dicatat di `model_only.json`. Tidak melakukan konversi dtype/quantization. File model-only tetap float32; optimizer upstream tidak dipakai untuk downstream.

Konfigurasi embedded dan bentuk tensor cocok: **12 layer Transformer, dimensi 768, FFN 3072, 12 head**, empat convolution stride/kernel 2, input 12 lead. Jumlah seluruh tensor parameter pretraining **90.883.072**, bagian encoder **90.373.248**. YAML publik justru 24/1024/4096/16; konfigurasi embedded yang digunakan. Hanya parameter masking/quantizer/projection SSL yang tidak dibutuhkan forward inference dikeluarkan saat strict encoder load; daftar key tersedia pada [inspection.json](../../assets/ecg_fm/584219ea492cdeef2e19ffbdf9c6ecc874ba427e/inspection.json).

## Hasil verifikasi yang sudah dijalankan Codex

| Verifikasi | Hasil |
|---|---|
| Hash/ukuran bobot upstream | Cocok dengan metadata rilis pinned |
| Salinan model-only | Semua tensor/config persis sama |
| Strict loading seluruh encoder | PASS; semua tensor finite |
| Input sintetis CPU 1250 sampel | Output `[1,78,768]`, error maksimum terhadap forward penulis **0** |
| Input sintetis CPU 2500 sampel | Output `[1,156,768]`, error maksimum **0** |
| Split 11 layer beku + layer terakhir | Identik dengan full forward CPU pada kedua panjang |
| Smoke sintetis bobot asli pada CPU | Forward/backward finite; layer terakhir menerima gradien |
| Tujuh test CPU sintetis | PASS; adapter, scaler, gradien, weighted effective batch, resume, grid/tie-break, end-to-end probe/tail + audit |
| Lima notebook baru | Sel kode valid dan **belum dieksekusi** |
| Persiapan metadata dan analisis kosong | PASS; 230 pasien/tiga kelas; **0/130 kasus asli** |
| Preservation V2–V9 | **1.915/1.915 file** cocok hash, tidak ada perubahan/error baca |
| Source numerik vendored | AST sama dengan upstream setelah mengabaikan import |

Resume CPU diuji dengan menghentikan fit sesudah dua epoch dan merusak CSV parsial. Pemulihan dari checkpoint menghasilkan history dan bobot bit-identik dengan fit empat epoch tanpa jeda. End-to-end sintetis menguji sembilan skenario, pemilihan inner, refit fixed-last, pasangan probe identik, skip kasus terverifikasi, dan tail satu fold. Scope tail di luar fold1/seed42 ditolak. Semua data uji tersebut dibuat acak/sintetis; skornya bukan hasil penelitian ECG.

Log test berada di `tmp/v10_synthetic_tests.log`. Smoke nyata CPU memakai data sintetis tercatat di `results/research_v10/20261006_ecg_fm_transfer_pilot_v10/synthetic_smoke_cpu.json`. Hash preservation V2–V9 dicatat terpisah di `tmp/v10_preservation_after.json`.

Bukti gabungan tersedia pada `results/research_v10/20261006_ecg_fm_transfer_pilot_v10/implementation_validation.json`. Ruang kosong pada pemeriksaan terakhir sekitar **5,57 GiB**; angka dapat berubah karena aktivitas sistem/iCloud. Tahap manual tetap melakukan pemeriksaan cadangan ruang sendiri.

## Desain yang disiapkan, belum hasil

Seluruh 230 pasien development dipertahankan. Window native diadaptasi 2000 → 500 Hz, 5000 → 1250 sampel, dan z-score per window/per lead. Interpolasi meniru normalizer penulis, bukan tambahan filter antialias. Sebelum klaim fisiologis perlu ablation resampling dan amplitude normalization. Durasi 2,5 detik berbeda dari segmen pretraining 5 detik. Pooling mengambil mean token, kemudian mean seluruh window pasien. Frozen extractor tidak di-fit pada pasien evaluasi; scaler fitur, class weight, head, serta layer terakhir dilatih hanya pada partisi training yang ditentukan.

Inner split mengikuti V8/V9. Tiap probe menjalankan 9 kombinasi LR/batch per fold pada seed 42, memilih konfigurasi/epoch dari inner macro-F1, lalu refit head baru untuk seed 42/43/44 dan menilai outer. Tail hanya fold1/seed42, dengan head LR dari pretrained probe dan grid LR tail sendiri. Jumlah fit: **90 inner + 30 outer frozen probe + 9 inner + 1 outer tail**. Tidak ada endpoint final/test V10. Seluruh seed head memakai backbone/split yang sama; variasi encoder/pretraining seed belum diukur.

V10 memilih hiperparameter di inner dan tidak memakai outer callback. Namun metode/input ini dirancang sesudah banyak analisis development sebelumnya, sehingga keseluruhan penelitian **belum merupakan estimasi confirmatory yang sepenuhnya nested**. Nilai macro-F1 V8/V9 tidak langsung menjadi causal control manfaat pretraining karena arsitektur/input/pooling berbeda. Random frozen probe juga tidak menggantikan training from-scratch penuh.

## Keterbatasan sumber pretraining yang ditemukan

ECG-FM memakai PhysioNet 2021 termasuk Ningbo. Paper asal lokalisasi juga memakai Ningbo First Hospital. Crosswalk pasien belum tersedia; keberadaan sumber rumah sakit yang sama **tidak membuktikan** waveform/pasien overlap, tetapi independensinya belum dapat disahkan. Metadata V10 menyimpan `pretraining_patient_independence_verified: false`. Piloting dibatasi eksploratif. [ECG-FM](https://arxiv.org/html/2408.05178v2), [paper lokalisasi](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2022.809027/full), [PhysioNet 2021](https://physionet.org/content/challenge-2021/1.0.3/).

Untuk klaim generalisasi publikasi: audit independensi pretraining dengan crosswalk/sumber asli atau checkpoint dari kohort independen yang dapat diaudit, tambahkan fine-tuning lintas fold dengan kontrol from-scratch setara, laporkan ketidakpastian per pasien dan evaluasi eksternal berlabel anatomi. Review ECG ahli belum ada; interpretasi model tetap komputasional. Historical test 59 pasien telah diakses sebelumnya. Peningkatan skor dan kelayakan Q1 belum diketahui.
