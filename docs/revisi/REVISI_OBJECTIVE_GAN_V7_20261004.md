# Revisi objective GAN V7 — 4 Oktober 2026

**Pembaruan 5 Oktober 2026:** training manual notebook 37–40 telah selesai dan seluruh lima fold PASS setelah audit native. [Analisis hasil dan batas klaim](ANALISIS_HASIL_V7_20261005.md). Desain di bawah dipertahankan sebagai protokol sebelum training; endpoint classifier masih tahap berikut.

**Pertanyaan eksperimen:** apakah pencocokan distribusi fitur kelas dari pasien GAN-fit memperbaiki fidelity sintetis dibanding objective moment minibatch V6, dengan generator, discriminator, data dan anggaran yang sama?

V6 fold 1 selesai 100 epoch dan menghasilkan 192 sampel sintetis, tetapi gagal pada 9/15 kombinasi kelas–keluarga. Roughness dan korelasi gagal pada seluruh kelas; amplitudo gagal pada kedua RVOT. Loss moment mulai mendatar, sehingga penambahan epoch saja belum mempunyai bukti sebagai perbaikan. [Diagnosis lengkap](ANALISIS_PILOT_GAN_V6_20261004.md).

## Dasar metode dan batas klaim

[Yamamoto, Song dan Kim, Parallel WaveGAN, ICASSP 2020](https://arxiv.org/abs/1910.11480) menggabungkan objective adversarial dan spectrogram dengan beberapa resolusi untuk sintesis speech. Ini mendukung mempertimbangkan informasi frekuensi pada training waveform. **V7 bukan implementasi/reproduksi Parallel WaveGAN:** data ECG tidak mempunyai pasangan target waveform untuk setiap latent, sehingga V7 memakai distribusi daya band pada setiap kelas, bukan loss rekonstruksi pasangan real/fake. Paper tersebut tidak membuktikan bahwa V7 akan memperbaiki lokalisasi ECG.

Operasi CPU/FFT dan gradien menggunakan API PyTorch standar. [Autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html), [Tensor.cpu](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html), dan [rfft](https://docs.pytorch.org/docs/stable/generated/torch.fft.rfft.html) menjadi rujukan implementasi; versi yang benar-benar dipakai dicatat dari environment project. Compatibility CPU→MPS tetap diuji melalui smoke manual pada notebook 37, tanpa mengandalkan fallback operasi otomatis.

Pencocokan quantile fitur classwise, normalisasi IQR dan bobot keluarga adalah **hipotesis engineering proyek** untuk masalah yang ditemukan, bukan novelty/validasi klinis yang telah dibuktikan. Komponen loss diuji sebagai satu paket terlebih dahulu; hasil positif belum menunjukkan kontribusi tiap komponen. Ablation komponen dan seed GAN tambahan tetap pekerjaan lanjutan bila paket ini menjanjikan.

## Kontrol yang dipertahankan

- Generator/discriminator **menggunakan kelas model V6 yang sama**, bukan arsitektur baru. Native 2000 Hz/5000 sampel dan delapan → 12 lead tetap.
- Fold/split inner/scalar/daftar window referensi disalin dengan hash, bukan dihitung ulang memakai pasien lain. Target loss hanya GAN-fit, bukan referensi QC.
- Seed 42 (+100×fold), inisialisasi dari awal, sampling/RNG, 100 epoch, 4 draw × jumlah pasien fit per epoch, batch 16, LR 1e-4, Adam/betas (0, 0.9), D:G 1:1, clip 5 sama.
- Semua 192 sampel sintetis/fold dipertahankan. Tidak menambahkan low-pass setelah generasi, mengubah smoothing preprocessing atau memilih sampel terbaik.
- Kode fitur/QC dan seluruh threshold sama. PASS membutuhkan semua keluarga/kelas memenuhi gate serta numeric/copy/diversity lolos. Tidak membaca waveform outer validation classifier atau test.
- V6 mempunyai pembanding lengkap pada fold 1 saja. Perbandingan angka setelah V7 selesai dilakukan pada fold 1; tidak mengklaim pasangan penuh 5 fold sebelum tersedia.

## Objective baru yang dibekukan sebelum training V7

Target log RMS, log span p95−p05, log roughness lag 1/4/16/64, 66 korelasi lead dan log daya relatif band dibentuk dengan bobot pasien sama dari tiga window deterministik/pasien. Quantile target 0,10/0,25/0,50/0,75/0,90; tidak diperbarui selama training. Quantile minibatch sintetis dibandingkan dengan target sesuai kelas, lalu dinormalisasi IQR fit dengan floor tetap agar fitur yang hampir konstan tidak menghasilkan bobot tak terbatas.

Tiga ukuran Welch: 500/1000/2000 sampel, overlap 50%, resolusi 4/2/1 Hz. Band: 0,5–5/5–15/15–40/40–100/100–250/250–500/500–1000 Hz. Daya di-log dengan floor 1e-6; ini tidak berarti floor/threshold klinis. Statistik dihitung pada CPU dengan graph gradien tetap terhubung, generator/discriminator dapat pada MPS. Biaya transfer/CPU dicatat.

Loss baru = adversarial + amplitude + roughness + correlation + spectrum, masing-masing keluarga berbobot 1. Semua bobot adalah pilihan pertama yang dicatat, belum optimal. Skala IQR minimum masing-masing 0,25/0,25/0,10/0,50. Formula, target, data, runtime dan kode terikat signature V7. Label/kohort dipertahankan.

## Urutan dan keputusan

1. **37:** prepare target fit-only dan smoke disposable. Tidak memulai training asli secara otomatis.
2. **38:** pilot fold 1 dari awal 100 epoch, generasi, QC yang sama.
3. **39:** perbandingan V6/V7, fraksi fitur gagal, jarak distribusi, copy/diversity, waveform dan biaya.
4. Jika **PASS**, setelah audit hasil pilot lanjut 40 untuk fold 2–5. Jika **FAIL**, laporkan perbaikan parsial/kelemahan dan tentukan perubahan berikut pada run baru; tidak melonggarkan gate.
5. Classifier K4 real-only/kontrol augmentasi/GAN baru setelah mutu per fold memenuhi syarat. V7 belum menyediakan endpoint classifier/final/test.

Penggunaan keluarga fitur yang mirip pada loss dan QC dimaksudkan menguji fidelity komputasional pada pasien referensi terpisah, bukan validasi semua struktur fisiologis. Mode collapse, memorisasi, label sintetis yang tidak benar, dan manfaat classifier tetap membutuhkan bukti masing-masing. Dataset diagnosis valid tidak memberi anotasi PVC pada setiap window.

Recipe dipilih setelah melihat kegagalan V6 pada development. Ini ablation eksploratif, bukan nested performance estimate. Untuk memperkuat naskah, hasil negatif V6 dipertahankan; manfaat positif harus konsisten pada perbandingan classifier dan evaluasi pasien berikutnya. Test historis tidak memilih recipe/seed/arm.

## Verifikasi implementasi

Sembilan tes sintetis menguji protokol/arsitektur kontrol, spektrum Torch versus SciPy, definisi quantile yang konsisten, gradien setiap loss dan respons terhadap noise 150 Hz, bank berbobot pasien tanpa membaca non-fit, resume identik pada CPU, gate fold 2, alur native termasuk plot pembanding, dan notebook valid. MPS smoke serta training V7 data asli belum dijalankan oleh Codex.

Manifest V7 mereferensikan fingerprint V6, data/preparation/checkpoint/sintesis/QC pembanding, kode baru, runner baru dan runtime. Kode/notebook/checkpoint/QC V6 tidak diubah. [Panduan eksekusi](PANDUAN_MANUAL_V7.md).

Persiapan data asli kelima fold selesai dengan nol langkah optimizer GAN. Semua enam artefak split/skala/fitur per fold identik dengan V6, bank target finite, dan 163 file terlindungi tidak berubah. Rincian audit: [`implementation_audit.json`](../../results/research_v7/20261004_fit_distribution_gan_v7/implementation_audit.json). Fingerprint V7: `6e6a9308d65a3160a17aab54e09ea6149d947ceca72632001a0e62f6bfd11125`.
