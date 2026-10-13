# TA GUWEH — project sharing copy (not a complete mirror)

This archive contains the actual project source tree copied read-only from the authorized Mac, with original relative paths beneath `TA GUWEH/`. No Mac writes, installs, training, or external upload were performed.

## Included
- Original root research scripts/READMEs, src, configs, scripts, research_v2–v14 and recovery/compatibility modules, analysis modules, tests, and notebooks.
- Original Markdown research documentation under docs (binary proposal/reference documents omitted because embedded patient information was not audited).
- V13 main three-class baseline/fine-tuning experiment text artifacts and V13 secondary binary/ensemble results; V14 binary CV and final-test text artifacts, including aggregate metrics, training histories/configs/protocols and anonymized patient-level predictions, preserving their original paths.
- Canonical real model-input arrays at `results/research_v2/20260930_anatomy_fs2000_v2/data/{development,test}/X.npy`; waveform bytes unchanged. Development: 230 patients / 2274 windows; test: 59 patients / 533 windows. Shape 12 x 5000 per window, float32. Row order unchanged.
- Original-path metadata/segments and fold manifests with consistent anonymous IDs. Five original fold partitions verified against the prior authorized anonymous export, not resplit.
- Source run protocol/selection/patient/fold metadata for V2/V3/V4/V7 where available; assets/ecg_fm config/download/inspection metadata, not the large pretrained weights.
- Added PACKAGING_DEPENDENCIES.txt records 134 installed source-Mac distributions by reading METADATA, not executing the project. It is not a tested portable requirements lock; no original top-level requirements/pyproject lock was found in the inventoried source.

## Privacy modifications
`hospital_id` column/key names retained for schema compatibility, but values are ANON0001 etc., NOT hospital IDs. Mapping was constructed only in process memory by verifying identical row-wise labels/start/length against the previously sanitized dataset; no mapping is exported. Source patient lists omit 45 Summit/unknown cases outside the 289-patient model cohort. All 334 known source patient-ID tokens scanned and absent from packaged text. Source signal-hash dictionary filenames also anonymized. Known patient-ID literals in code/docs replaced and notebook outputs cleared; per-file manifest distinguishes modified files. Noncohort identifier-bearing metadata that could not be safely retained is excluded. No Diagnosis.xlsx, raw ECG files, clinical identity mapping, credentials, or .env.

Anonymized physiological signals are still sensitive; authorized supervisor sharing only, not public redistribution. Consent/ethics/license were not independently verified.

## Excluded / limitations
Not a complete mirror. Excludes .git/.venv/node_modules, caches/temporary/chart/slide-build folders, raw and duplicate processed ECG datasets, Diagnosis.xlsx, external downloadable corpora, older large experiment result trees (except the selected ancestor metadata), binary reports/images/PDFs, checkpoints/model weights, synthetic GAN arrays, locks and other unneeded binary artifacts. Precise included paths, modification labels and excluded per-top-level counts/sizes are in PACKAGING_MANIFEST.json. Excluded filenames that themselves contain patient identifiers are deliberately not listed.

Full training/inference reproducibility is NOT claimed: GAN caches/checkpoints and predecessor result artifacts are omitted; existing experiment signatures incorporate original IDs/metadata hashes/runtime and will reject sanitized metadata under the frozen old run ID. Historical protocol hashes retained as provenance, not silently rewritten. Data loaders can read the schema-compatible canonical cache, but frozen end-to-end context requires explicit adaptation/new run IDs, missing dependencies/artifacts and a suitable environment. No training or project code was run. No need to configure Google Drive: upload this archive manually.

PACKAGING_SHA256SUMS.txt hashes every included file except itself. ZIP CRC and every archived file SHA256 are verified on VPS after creation.
