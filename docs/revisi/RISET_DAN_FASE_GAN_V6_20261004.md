# Arah penelitian dan protokol GAN V6 — 4 Oktober 2026

**Tujuan berikut: menguji apakah augmentasi GAN yang menjaga amplitudo dan hubungan lead membantu lokalisasi tiga kelompok pada evaluasi pasien.** Fase pertama yang diimplementasikan adalah mutu GAN, notebook 33–36. Baseline K4 dipertahankan setelah hasil negatif V5. Skor 0,9 dan penerimaan Q1 belum dapat dijanjikan; kontribusi harus ditunjukkan melalui pembanding, analisis kegagalan dan evaluasi yang dapat direproduksi.

## Dasar dari referensi primer yang diperiksa

| Referensi | Pelajaran untuk desain ini | Batas penerapan |
|---|---|---|
| [Golany dkk., SimGANs, ICML 2020](https://proceedings.mlr.press/v119/golany20a.html) | Sintesis berbasis pengetahuan dinamika jantung diuji untuk augmentasi klasifikasi | V6 tidak mengimplementasikan simulator ODE SimGANs; bukan reproduksi SimGANs |
| [Thambawita dkk., DeepFake ECG, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8578227/) | Delapan kanal independen diturunkan menjadi 12 lead; evaluasi sinyal melampaui loss adversarial | Paper memakai ECG normal 500 Hz/10 detik dan evaluator MUSE 12SL. V6 memakai 2000 Hz/2,5 detik, kondisi diagnosis asal, tanpa evaluator klinis tersebut |
| [Miyato dkk., SN-GAN, ICLR 2018](https://arxiv.org/abs/1802.05957) | Spectral normalization discriminator sebagai pilihan stabilisasi komputasional | Penelitian asli pada citra; stabilitas/kinerja ECG proyek tetap hipotesis |
| [Miyato dan Koyama, projection cGAN, ICLR 2018](https://arxiv.org/abs/1802.05637) | Label dimasukkan melalui projection discriminator | Generator conditional tidak otomatis membuktikan kebenaran label anatomi sintetis |
| [Liu dkk., CAE-WaveGAN, 2025](https://www.nature.com/articles/s41598-025-20470-3) | Analisis arsitektur, frekuensi/morfologi dan pembanding 12 lead | Tugas berbeda dari lokalisasi tiga kelompok. V6 tidak memakai skor paper sebagai target akurasi atau SSIM sebagai bukti klinis; contoh visual ditentukan sebelum melihat hasil |
| [TRIPOD+AI, BMJ 2024](https://www.bmj.com/content/385/bmj-2023-078378) dan [PROBAST+AI, BMJ 2025](https://www.bmj.com/content/388/bmj-2024-082505) | Acuan pelaporan model prediksi dan penilaian kualitas, bias, serta applicability | Panduan ilmiah; bukan ambang skor atau jaminan jurnal Q1 |

Referensi proposal yang terverifikasi tetap ada di `REFERENSI_TERVERIFIKASI_INTI.bib`; tambahan V6 di `REFERENSI_METODE_V6.bib`. Referensi yang metadata/implementasinya belum jelas tidak dijadikan bukti.

## Mengapa GAN lama perlu diganti

GAN lama memakai asumsi/label historis, skala 5 dengan clipping dan output tanh, BatchNorm discriminator, serta exposure berbasis seluruh segmen. Diagnosis sebelumnya menemukan sintetis jauh lebih kasar. Audit tambahan pada **49 window outer-training fold 1 saja** mendapatkan sekitar **0,5714%** nilai setelah scaler classifier melewati ±5; clipping lama mengubah sebagian amplitudo. Ini audit sampel, bukan proporsi seluruh dataset.

Audit yang sama pada cache sebelum scaler mendapatkan median RMS residual aljabar relatif terhadap RMS enam limb lead: **0,01782 (III), 0,00597 (aVR), 0,01133 (aVL), 0,00883 (aVF)**. Cache memakai denoising per lead sehingga hubungan tidak persis nol. Residual kecil mendukung mencoba parameterisasi delapan lead; ini keputusan engineering, bukan validasi klinis.

Persamaan yang digunakan: **III = II − I; aVR = −(I + II)/2; aVL = I − II/2; aVF = II − I/2**. Generator mempelajari I, II, V1–V6 pada skala global yang sama dengan output linear, lalu menurunkan empat lead. V6 dicatat sebagai recipe baru, sehingga hasil historis tidak dicampur.

## Recipe pilot yang ditetapkan sebelum training

- **Data:** 12 × 5000 sampel, 2000 Hz; label LVOT / RVOT cusp / RVOT septal-freewall-other tetap. Cache dan manifest outer fold V4 digunakan ulang.
- **Pemisahan:** sesuai manifest tetap, 183–185 pasien outer-training menjadi 146–148 GAN-fit dan 37 referensi mutu, stratifikasi diagnosis. Fold1: 147 fit/37 referensi/46 outer validation. Skala GAN hanya fit pada pasien GAN-fit fold itu. Waveform 45–47 outer-validation dan 59 test tidak dibaca oleh GAN/QC pada fold yang sedang dikerjakan.
- **Generator:** latent 64 + embedding 16, linear ke 64 × 625; tiga upsampling linear ×2 dengan Conv1d/GroupNorm/ReLU; output delapan lead linear lalu 12 lead aljabar. Panjang native dipertahankan.
- **Discriminator:** empat spectral-normalized Conv1d, global mean dan projection label; hinge loss, tanpa BatchNorm/sigmoid. Ini adaptasi komponen metode, bukan reproduksi model paper persis.
- **Loss generator:** adversarial + 1 × moment matching classwise untuk log RMS, roughness lag 1/4/16/64 dan korelasi lead; mean serta SD minibatch. Komponen tambahan adalah hipotesis engineering untuk masalah roughness lama, perlu ablation setelah pilot bermutu.
- **Training:** 100 epoch tetap, batch 16, Adam LR 1e-4/betas (0, 0.9), update D:G = 1:1, gradient clip 5. Sampling uniform kelas → pasien → window; anggaran total empat kali jumlah pasien fit per epoch. Memakai checkpoint epoch terakhir, tanpa seleksi dari F1 classifier atau loss GAN.
- **Sintesis:** 64 window per kelas/fold, seed terpisah tetap, bukan pasien baru. Semua sampel dipertahankan untuk audit; tidak hanya memilih gambar terbaik.
- **Mutu:** finite, constant/copy/diversity, amplitudo, roughness, Welch PSD 0,5–1000 Hz, korelasi lead, residual aljabar dan nearest neighbour full native terhadap seluruh window fit. Bobot real per pasien sama; ambang QC ditetapkan sebelum training.

Satu seed GAN per fold digunakan untuk **pilot engineering**, belum mengukur variasi generator. Jika manfaat classifier menjanjikan, tambah seed GAN untuk memisahkan variasi generator dari seed classifier. Seratus epoch bukan batas maksimal; perpanjangan/recipe baru perlu dicatat pada run baru sebelum dibandingkan.

## Fase untuk memperkuat peluang publikasi

| Prioritas | Pertanyaan penelitian | Bukti yang harus dihasilkan | Status |
|---|---|---|---|
| 1 | Apakah GAN menjaga informasi sinyal dan tidak sekadar menyalin? | Screening per kelas, spektrum, 12 lead, copy/diversity, biaya dan kegagalan | **V6 siap manual**: 33–35 pilot; 36 setelah PASS |
| 2 | Apakah GAN membantu classifier melebihi real-only dan variasi real biasa? | Kontrol K4 berpasangan, 5 fold × 3 seed, macro-F1 pasien, recall kedua RVOT, OOF dan waktu | Setelah pilot; endpoint classifier belum dibuat |
| 3 | Komponen mana yang membantu? | Ablation moment loss/parameterisasi lead atau CNN–Transformer; seed GAN tambahan bila positif; tiga arsitektur proposal sebagai pembanding | Rencanakan setelah fase 2; ubah faktor secara terkontrol |
| 4 | Apakah pretraining dan fine-tuning membantu? | Sumber bobot pretrained, pemisahan data pretraining, pembanding from scratch, recipe dan biaya | Belum diimplementasikan; tuning LR/batch dari bobot acak belum merupakan fine-tuning bobot pretrained |
| 5 | Apakah metode dapat digeneralisasi? | Outer scoring pasien dengan recipe/epoch dipilih hanya di inner; kohort baru dengan label asal kompatibel; uncertainty/bootstrap pasien | Belum tersedia untuk recipe baru; CV lama tidak menjadi nested hanya dengan mengganti nama |
| 6 | Apakah naskah bisa direproduksi dan kontribusinya jelas? | Bab III–V, tabel positif/negatif, alur cohort, kontribusi faktual, model card, pelaporan metode | Bisa mulai sekarang dari hasil V2–V5 yang terverifikasi |

Perbandingan classifier tidak boleh menambahkan ribuan sintetis ke loss seolah-olah pasien independen. Pertahankan batch pasien 8, K4, recipe, scaler, seed dan fold; tetapkan porsi augmentasi serta kontrol real dengan anggaran update/exposure yang sebanding. Bila mengganti window dalam bag, catat sebagai bag artificial, bukan rekaman pasien nyata. Asumsi identitas pasien dalam MIL dan efek pooling perlu dibahas.

Fase classifier membutuhkan protokol tersendiri setelah mutu pilot dipahami. Jika GAN gagal atau manfaat tidak konsisten, hasil negatif tetap dilaporkan. Alternatif berikutnya ialah pretraining atau representasi beat yang lebih sesuai sumber, dengan kandidat R otomatis tetap tidak dianggap PVC terverifikasi. INCART mendukung audit detector, bukan validasi asal tiga kelas. PTB-XL tidak otomatis mempunyai label origin kompatibel; mungkin relevan untuk pretraining dengan penjelasan domain/sampling.

Banyak eksperimen telah dilakukan pada development dan test historis sudah diamati. Fase baru tetap eksploratif pada development. Membagi ulang pasien yang sama atau mengulang seed test tidak menghasilkan kohort unseen baru. Estimasi generalisasi kuat memerlukan pipeline seleksi dibekukan dan evaluasi independen. Tanpa ahli ECG, hasil komputasional serta batas anotasi ditulis jelas, tanpa klaim validasi klinis sintetis.

## Pekerjaan yang selesai pada implementasi ini

- Kode V6, protokol terpisah, manifest data/runtime/code, runner manual dan notebook 33–36 tersedia.
- Sembilan tes sintetis lulus: isolasi pasien, skala fit-only/tanpa clipping, relasi lead, resume identik CPU, penolakan copy/collapse/nonfinite/noise, gate fold 2, seluruh endpoint/visualisasi, serta notebook valid.
- Persiapan development **telah selesai untuk lima fold**, tanpa optimizer GAN: 146–148 pasien fit, 37 referensi mutu, 45–47 outer validation per fold. [Tabel lengkap](PANDUAN_MANUAL_V6.md#pembagian-pasien-dan-sinyal). MPS smoke dan training data asli tetap Anda jalankan manual; mutu sintetis dan skor classifier V6 belum tersedia.
- Hash 83 file kode/config/notebook V2–V5 yang dipantau sebelum/sesudah implementasi tetap identik. Fingerprint dan hasil kontrol V4/V5 diverifikasi saat konteks V6 dibuka.

Urutan eksekusi ada pada [panduan V6](PANDUAN_MANUAL_V6.md). Hasil yang dicari adalah bukti manfaat metode dan batasnya yang dapat dinilai reviewer.
