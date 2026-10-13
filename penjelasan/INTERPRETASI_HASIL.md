# Membaca hasil tanpa mengganti endpoint

Sumber ringkasan: [README diskusi](../hasil_eksperimen/README.md), [CSV endpoint](../hasil_eksperimen/comparison_endpoints.csv), [CSV subregion](../hasil_eksperimen/comparison_subregions.csv). Semua angka adalah hasil historis, bukan eksekusi ulang pada VPS.

CV 3class fine-tuning: Encoder 0,6531±0,0625; PatchTST 0,6403±0,0590; Hierarchical 0,6418±0,0579. Test historis mean 5 seed masing-masing 0,5722±0,0693; 0,5646±0,0582; 0,5614±0,0653. Ensemble 3class masing-masing 0,6082; 0,5843; 0,5185. Jangan menggunakan mean sebagai ensemble, SD sebagai confidence interval, atau seed terbaik sebagai hasil utama.

Biner CV fine-tuning V13→V14: Encoder 0,8641→0,8903 (Holm p≈0,0878), PatchTST 0,8553→0,8992 (≈0,0088), Hierarchical 0,8546→0,8806 (≈0,0536). Hanya PatchTST lolos aturan. Test PatchTST mean V13 0,8383±0,0181 vs V14 0,8484±0,0406; ensemble keduanya F1=0,8659, AUC=0,9619, accuracy=0,8983. ΔF1 ensemble=0, McNemar p=1, bootstrap CI [0,0]. Kenaikan CV tidak menjadi bukti kenaikan ensemble test.

Macro-F1 adalah rata-rata F1 kelas tanpa bobot support. Sensitivitas kelas adalah recall kelas, bukan nilai macro umum. CV 15 fold×seed bukan 15 kohort independen. Test 5 seed mengevaluasi 59 pasien yang sama. Ensemble dirata-ratakan probabilitasnya dahulu, baru dihitung skor.

Holdout pernah dilihat historis, bukan untouched/external test. Gagasan biner muncul setelah hasil test V4. Seleksi grid dan estimasi CV memakai CV yang sama, bukan nested outer evaluation. Praregistrasi berikutnya tidak menghapus paparan historis. Aturan klinis otomatis belum divalidasi ahli; CI delta AUC melintasi nol. Jangan klaim clinical superiority, universal winner atau bebas leakage tanpa audit independen. Window berlabel origin belum berarti beat PVC tervalidasi ahli.
