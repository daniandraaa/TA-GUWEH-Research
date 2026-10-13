# Addendum metodologi: eksperimen V3 tanpa anotasi ECG baru dari ahli

Draft untuk dibahas dengan pembimbing. Menjelaskan implementasi yang sudah disiapkan, bukan hasil eksperimen V3 atau naskah final yang sudah disetujui.

## Definisi target dan sumber data

Penelitian menggunakan waveform ECG 12 sadapan dan label lokasi ablasi pada dataset publik Ningbo. Label penelitian dikelompokkan menjadi LVOT, RVOT cusp dan RVOT septal/free-wall/other. LC, AC, RC adalah cusp pada RVOT; istilah Non-OT pada kode/proposal historis dikoreksi tanpa mengubah keanggotaan pasien. RVOTOther dipertahankan pada kelompok ketiga sebagai kategori residual sumber, tanpa menetapkan subanatomi yang tidak tersedia. [Paper sumber tugas](https://pmc.ncbi.nlm.nih.gov/articles/PMC8962834/).

Kohort lokal berisi 334 pasien, sesuai jumlah pada deskripsi dataset publik 2020, dengan 289 pasien yang dapat dipetakan ke endpoint tiga kelompok; 5 Summit dan 40 unknown tidak masuk target ini. Pembagian historis 230 pasien development dan 59 test dipertahankan. Dataset asal artikel 2022 memuat kohort lebih besar dan skema klasifikasi berbeda, sehingga skor sumber bukan baseline numerik untuk eksperimen lokal. [Deskripsi dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC7090065/).

Label anatomi tingkat pasien berasal dari metadata sumber. Tidak tersedia anotasi PVC per sampel yang dapat ditelusuri di project dan tidak tersedia ahli ECG untuk membuat anotasi baru. Oleh sebab itu, jendela deteksi otomatis disebut **kandidat R/QRS**, bukan PVC terverifikasi. Penelitian ini tidak mengklaim telah melakukan adjudikasi klinis beat atau memvalidasi diagnosis baru.

## Masukan dan ablation

Sampling mengikuti sumber 2000 Hz. Preprocessing berupa high-pass 0,5 Hz dan denoising coif5 level5 dengan threshold SURE soft. Sinyal yang direkonstruksi tetap berupa waveform waktu; tidak disebut koefisien DWT sebagai masukan jaringan. Filter/deteksi dilakukan independen per rekaman tanpa pemanfaatan label asal. StandardScaler dipasang hanya pada pasien training masing-masing fold.

Empat arm menguji representasi window 2,5 detik versus kandidat 0,5 detik dan kontribusi sampling per segmen versus per pasien. Crop kandidat menggunakan 200 ms sebelum/300 ms setelah ekstremum yang disepakati dua sadapan terpilih. Detektor engineering menggunakan derivative-energy pada sinyal bandpass 5–25 Hz, consensus tolerance50 ms, refractory250 ms, dan maksimal64 crop per pasien. Aturan ditetapkan sebelum melihat hasil CV V3.

Semua pasien tetap memiliki masukan. Bila tidak ditemukan kandidat yang lengkap, rekaman mendapat window fallback 500 ms dengan flag eksplisit; tidak ada pasien yang dibuang karena kesulitan detektor. Jumlah/keberadaan fallback, agreement sadapan dan candidate rate dilaporkan. Quality flags menunjukkan kasus yang perlu inspeksi teknis, bukan label penyakit atau keputusan mengubah diagnosis.

Sampling dengan pengembalian mengambil jumlah draw sama dengan jumlah window2,5 s dari training fold, agar jumlah langkah optimizer sama pada empat arm untuk batch size yang sama. Sampling per pasien menggunakan bobot1/n crop pasien sehingga setiap pasien memiliki massa probabilitas yang sama. Loss antar-kelas memakai inverse jumlah **pasien** setiap kelas, dihitung hanya pada training fold. Mean dan scale tetap dihitung dari seluruh potongan training pada representasi tersebut, sehingga keseimbangan sampler tidak membuat scaler menjadi patient-weighted.

## Model, tuning dan seleksi

Model neural terdiri atas CNN-Transformer Encoder, CNN patch transformer yang terinspirasi PatchTST, dan CNN Hierarchical Transformer. Model lokal merupakan adaptasi dan dilatih dari awal; tidak dinyatakan sebagai reproduksi paper asal atau fine-tuning pretrained. V3 CNN patch mempertahankan sisa feature sequence melalui padding dan pooling berdasarkan panjang valid. Encoder dan Hierarchical tetap memakai arsitektur V2. Perubahan patch tidak dipakai berbeda-beda antarskenario V3.

Grid LR[1e-4,5e-4,1e-3] dan batch[16,32,64] menghasilkan9 kombinasi. Tiap kombinasi dievaluasi pada5 fold pasien yang sama untuk seluruh arm/model. Early stopping berdasarkan macro-F1 validation yang dihitung setelah probabilitas seluruh potongan dirata-ratakan per pasien. Primary metric ialah macro-F1 tiga kelas; accuracy, macro precision/recall/specificity, AUC one-vs-rest, confusion matrix dan per-class recall dilaporkan bersama jumlah pasien.

Baseline ExtraTrees menggunakan fitur waveform deskriptif/bandpower tanpa inferensi koordinat atau aturan klinis yang belum divalidasi. Dua representasi dievaluasi pada manifest pasien yang sama;9 konfigurasi max_features/min_samples_leaf,500 pohon,max_depth12. Bobot pasien/kelas ditentukan dari training fold. Baseline ini dinyatakan sebagai comparator baru, bukan reproduksi fitur klinis paper sumber.

Parameter/arm dipilih dari mean macro-F1 development CV, kemudian SD sebagai tie-break. Model neural utama dipilih dari CV; comparator klasik dilaporkan terpisah, termasuk bila mengungguli neural model. Epoch final berasal dari median best epoch fold terpilih. Lima training final memakai seed42–46 pada development penuh dan hasil primer tetap seed42. Tidak ada pemilihan model/parameter/seed/ensemble berdasarkan test.

CV terpilih mengandung bias seleksi karena fold yang sama dipakai mencari konfigurasi. Hasil itu tidak disebut estimasi nested CV atau bukti generalisasi eksternal. Lima seed tidak menjadi lima cohort independen. Test historis sudah diamati pada eksperimen terdahulu, sehingga diberi nama historical test; mengganti split atau seed tidak menjadikannya data eksternal.

## Interpretabilitas dan evaluasi detektor

SHAP V2 menjelaskan checkpoint utama V2 seed42 dengan final scaler V2. Background hanya berasal dari development, satu segmen/pasien yang dipilih secara acak. Sampel penjelasan ialah satu segmen tengah untuk tiap pasien test terpilih, distratifikasi per kelas. Expected gradients menjelaskan true-class logit segmen, bukan mean probability agregat pasien. Attribusi waveform saling berkorelasi tidak membuktikan sebab fisiologis atau validitas klinis.

Tanpa anotasi baru dari ahli, detektor dapat diuji pada anotasi beat publik INCART menggunakan header sampling aslinya. Benchmark menghitung matching satu-ke-satu dengan tolerance±100 ms, precision/recall/F1 dan galat waktu, serta berapa beat beranotasi V yang terdeteksi. Tidak ada classifier PVC dilatih dalam benchmark tersebut. INCART tidak memiliki label lokasi ablasi tiga kelas lokal, sehingga bukan validasi eksternal lokalisasi. [Dataset INCART](https://physionet.org/content/incartdb/1.0.0/).

## Status GAN dan klaim penelitian

V3 ini menggunakan data asli. GAN historis tidak digunakan kembali pada preprocessing yang berubah. GAN yang dikoreksi dan ablation augmentasi tetap pekerjaan terpisah jika menjadi objektif wajib proposal. Generator/sampel sintetis hanya boleh belajar dari training fold; validation/test selalu asli. Pemeriksaan engineering relasi sadapan, amplitudo, spectrum dan keragaman tidak sama dengan review morfologi oleh ahli.

Tidak ada ground truth posisi dalam milimeter sehingga tidak dihitung MAE/RMSE koordinat. Heatmap SHAP menggambarkan kontribusi sadapan/waktu, bukan peta anatomi sumber PVC yang sudah tervalidasi. Akurasi training, klasifikasi biner eksploratif dan skor aritmia pada literatur tidak menggantikan hasil patient-level tiga kelas.

Kontribusi yang dapat diperiksa ialah pengaruh representasi dan keseimbangan pasien pada dua kelompok RVOT, kegagalan kandidat otomatis, perbandingan metode sederhana/deep learning dan ketidakpastian. Publikasi Q1 membutuhkan hasil empiris, kontribusi dan pelaporan yang dapat dinilai; penerimaan belum dapat dijanjikan dari rancangan ini.
