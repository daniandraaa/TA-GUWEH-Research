# Pra-registrasi pembanding klinis: kriteria EKG standar LVOT vs RVOT

Ditetapkan **2026-10-08 13:26:33 WIB**, sebelum kriteria dihitung pada data mana pun.

## Tujuan

Membandingkan model deep learning dengan aturan EKG yang biasa dipakai klinisi, pada pasien yang sama. Kriteria **tidak dilatih**: ambangnya diambil dari literatur dan tidak diubah setelah melihat hasil. Karena itu kriteria ini sah dievaluasi pada development (230) maupun test (59).

## Kriteria (ambang tetap)

| Kriteria | LVOT bila | Rujukan |
|---|---|---|
| Zona transisi prekordial (TZ) beat PVC | R ≥ S pertama kali di V1–V2. V3 tidak tegas; ≥ V4 berarti RVOT. | Praktik klinis umum |
| Indeks V2S/V3R | ≤ 1,5 | Yoshida dkk., 2014, *Heart Rhythm* |
| Rasio transisi V2: [R/(R+S)]PVC ÷ [R/(R+S)]sinus di V2 | ≥ 0,6 | Betensky dkk., 2011, *JACC* |

## Pengukuran otomatis

- Kandidat QRS dan amplitudo R/S per sadapan memakai `research_v11.morphology` dari rekaman mentah `PVCVTRawECGData` (2000 Hz). Pengukur ini **belum divalidasi ahli**; ini disebut sebagai keterbatasan.
- **Beat PVC** = kandidat dengan rentang QRS ≥ 120 ms; **sinus** = < 120 ms. Nilai per pasien = median beat terpilih.
- Bila tidak ada beat yang terukur, pasien dicatat **tidak dapat diputuskan**.

## Pelaporan

- **Cakupan:** proporsi pasien yang dapat diputuskan.
- **F1 paksa:** pasien yang tidak dapat diputuskan dianggap RVOT (kelas mayoritas).
- **F1 pada pasien tercakup.**
- **AUC dari skor kontinu** pada pasien yang terukur.
- Pembanding model: V13 fine-tuning. Development memakai prediksi validasi CV (OOF, rata-rata 3 seed); test memakai ensemble 5 seed. Ambang model tetap 0,5.
- **Uji:** selisih AUC (model − kriteria), bootstrap pasien terstratifikasi 5.000 kali (seed 20261008), pada pasien yang terukur kedua skornya.

## Kode yang dikunci (SHA-256)

- __init__.py: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- clinical.py: e2144c5d51e2a263097f66d9256ace116cb610a5bf27a52edd92abdfbec5bab4

Perlu diverifikasi: rujukan Yoshida 2014 dan Betensky 2011 dicek ulang ke naskah aslinya sebelum masuk paper.
