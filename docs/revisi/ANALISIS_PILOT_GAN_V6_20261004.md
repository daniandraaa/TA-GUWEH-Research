# Notebook 36 berhenti: pilot GAN V6 belum lolos mutu

**Training fold 1 selesai; notebook36 berhenti karena `quality.json` menyimpan `passed: false`.** Pesan `RuntimeError: Fold1 gagal screening mutu...` berasal dari `require_quality(study, 1)` pada sel prasyarat, sebelum training fold 2. Ini penghentian yang sesuai protokol, bukan kegagalan MPS, kehabisan memori atau checkpoint rusak.

Notebook 33–35 telah selesai: 4/4, 5/5 dan 4/4 sel kode, tanpa error tersimpan. Notebook 36 baru menjalankan setup dan sel prasyarat; fold 2–5 belum dimulai. Mengulang36 pada run yang sama akan berhenti lagi. Tidak perlu mengulang training 34, mengganti device, atau mengedit `passed` menjadi true.

## Hasil yang sudah selesai

- Smoke MPS dengan batch16×12×5000 lulus.
- GAN fold 1 selesai100 epoch;192 window sintetis,64 per kelas,2000 Hz/2,5 detik.
- Nilai numerik finite, lead tidak konstan, tidak ada exact-copy training atau near-copy yang memenuhi ambang screening ini.
- Seluruh 192 sampel mempunyai hash berbeda; screening near-duplikasi sintetis lolos pada setiap kelas.
- Relasi aljabar limb lead lolos pada ketiga kelas.

Hash dan keputusan screening dihitung ulang dari tabel fitur pada [audit terpisah](../../results/research_v6/20261004_conditional_gan_qc_v6/analysis_20261004_pilot_fold 1/REPORT.md). Bukti di `audit.json` pada folder yang sama. Kode analisis baru di `analysis_v6/pilot_diagnosis.py`; tidak mengubah kode, protokol, checkpoint, QC atau notebook V6 yang telah dieksekusi.

## Mengapa gate gagal

Angka berikut adalah **persentase fitur gagal screening**, bukan akurasi classifier. Batas protokol adalah maksimum 25% per kelas/keluarga.

| Kelompok | Amplitudo | Korelasi lead | Roughness | Spektrum | Limb residual |
|---|---:|---:|---:|---:|---:|
| LVOT | 25,0% — lolos | **34,8%** | **66,7%** | **35,7%** | 0% — lolos |
| RVOT cusp | **41,7%** | **51,5%** | **47,9%** | 19,0% — lolos | 0% — lolos |
| RVOT septal/freewall/other | **45,8%** | **45,5%** | **33,3%** | 22,6% — lolos | 0% — lolos |

**Sembilan dari 15 kombinasi kelas–keluarga gagal.** Screening spektrum lolos agregat pada dua kelas RVOT, tetapi beberapa fitur frekuensi tetap gagal; lolos agregat tidak berarti setiap band sudah cocok.

Contoh spesifik pada lead V2 kelas RVOT cusp, dibandingkan median referensi dengan bobot pasien sama:

- Rasio derivative1 sampel terhadap RMS: real 0,02278 vs sintetis 0,03549, sekitar**1,56 kali**. Ini menunjukkan variasi sampel lebih kasar pada ukuran tersebut.
- Daya relatif 40–100 Hz: real **0,0424%** vs sintetis **1,4386%**. Ini selisih distribusi frekuensi pada cache proyek; bukan ambang diagnosis klinis.
- Daya relatif 100–250 Hz: real sekitar0,000364% vs sintetis 0,1399%. Rasio besar dipengaruhi baseline real yang sangat kecil; perbandingan absolut perlu dilaporkan bersama rasionya.
- Korelasi V1–V2: real **0,9084** vs sintetis **0,3778**. Ketepatan empat persamaan limb lead belum menjamin struktur korelasi precordial sudah sesuai.
- Rentang p95−p05 V2 pada skala global: real **3,9687** vs sintetis **2,9025**. Distribusi amplitudo belum sesuai meskipun RMS beberapa lead mendekati real.

Waveform pada notebook 35 memperlihatkan variasi tajam dan ketidakselarasan bentuk antar-lead sintetis. Interpretasinya dibatasi pada pemeriksaan komputasional/visual; tanpa ahli ECG kita tidak menyatakan ada diagnosis/patologi sintetis tertentu atau bahwa kondisi anatominya benar.

## Training selesai belum berarti mutu sudah cukup

Moment loss rata-rata turun dari sekitar0,4238 pada epoch 1 menjadi0,0630 pada epoch 100. Rata-rata epoch 26–50, 51–75 dan 76–100 adalah 0,0674,0,0654 dan0,0658: perbaikan objective ini mulai mendatar, sedangkan distribusi referensi masih gagal. Loss minibatch tersebut tidak sama dengan mutu sinyal atau macro-F1 classifier.

Norm gradien sebelum clipping pada kedua jaringan melampaui ambang 5 untuk rata-rata norm per epoch pada seluruh 25 epoch terakhir. Ini menunjukkan clipping aktif, tetapi tidak membuktikan optimizer rusak dan bukan alasan otomatis menghilangkan clipping. Menambah epoch saja belum punya bukti akan memperbaiki kelemahan di atas.

## Langkah berikut yang diperlukan

1. **Pertahankan V6 sebagai hasil pilot negatif yang sah.** Tidak mengubah ambang, memilih hanya sintetis yang tampak bagus atau meneruskan fold 2–5 dengan generator ini.
2. **Revisi objective pada paket/run baru:** uji pengendalian distribusi spektrum dan roughness pada beberapa skala waktu, bukan hanya loss adversarial serta momen minibatch. Ini hipotesis perbaikan yang dipandu kegagalan, belum peningkatan yang terbukti.
3. **Periksa struktur bersama antar-lead dan distribusi amplitudo classwise.** Tambahan persamaan limb lead sudah bekerja; berikutnya korelasi precordial dan bentuk/distribusi amplitudo yang masih meleset. Bobot komponen loss perlu dirancang secara terkontrol, bukan menaikkan semuanya sekaligus.
4. **Ulang hanya pilot fold 1 pada recipe baru**, dengan pasien fit/reference yang sama dan screening tetap. Tidak membuka validation classifier atau test untuk memilih generator. Setelah lolos, barulah seluruh fold dan perbandingan classifier K4/kontrol augmentasi/GAN.

**Pembaruan setelah diagnosis:** recipe lanjutan kini tersedia pada paket/run V7 terpisah, dengan notebook 37–39 untuk pilot dan pembanding mutu; lihat [panduan V7](PANDUAN_MANUAL_V7.md). Training V7 data asli belum dijalankan. Notebook 36 V6 tetap terhenti secara benar. Sinyal sintetis bukan pasien baru; lolos copy/duplikasi juga bukan sertifikasi privasi atau validasi klinis. V6 belum menghasilkan skor classifier sehingga hasil ini tidak membuktikan performa lokalisasi sudah maksimal.
