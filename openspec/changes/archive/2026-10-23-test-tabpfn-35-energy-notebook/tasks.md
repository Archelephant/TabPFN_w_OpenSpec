# Tasks

Reference: `specs/smartplug-scenario-classification/spec.md` for what, `design.md` for how.

Note: runtime verification must happen in Google Colab (data lives in the user's Drive); static/file-level checks can be done locally at implementation time.

## 1. Notebook scaffold

- [x] 1.1 Create `TabPFN35_SmartPlug.ipynb` at the project root as a valid nbformat-4 notebook with Google-Colab metadata (kernelspec/accelerator as in `TimesFM3_DeepSeek.ipynb`), plus a title markdown cell in Russian explaining the goal (test TabPFN-3.5 on SmartPlug scenario data), the source links (arXiv 2609.17895, arxiviq review, PriorLabs/TabPFN, HF `Prior-Labs/tabpfn_3_5`), the two resource questions, and the non-commercial license note. Verify: the file parses as JSON (`python3 -c "import json; json.load(open(...))"`), opens in Jupyter/Colab.

## 2. Environment and dependencies

- [x] 2.1 Add the dependency cell: `%pip install tabpfn`, with a Russian comment explaining why only this package is installed and everything else comes preinstalled in Colab. Verify: no other pip/install line exists in the notebook; every later import maps either to `tabpfn` or a Colab-preinstalled package.
- [x] 2.2 Add imports cell (stdlib `os`, `re`, `time`, `warnings`; `numpy`, `pandas`, `torch`, `tqdm` only as actually used later) with a seed-setting line, with Russian comments per import. Verify: every imported name is used by a later cell (grep the notebook source), no unused import remains at review time.
- [x] 2.3 Add a markdown cell restating TabPFN-3.5's documented resource characteristics (~8 GB VRAM typical GPU, 16 GB only for very large datasets, CPU feasible up to ~5000 rows, no scaling / no one-hot needed, batch predict ≤1000 rows), then a code cell that prints `torch.__version__`, CUDA availability, GPU name and total VRAM, or an explicit "CPU only" line, and a verdict line answering GPU-vs-CPU and T4-fit for this dataset. Verify: on a T4 runtime the output names Tesla T4 with its VRAM; on a CPU runtime the CPU verdict prints; static review confirms documented limits appear in the markdown cell.

## 3. Data ingestion from Google Drive

- [x] 3.1 Add section markdown cell ("Загрузка и подготовка данных"), then a drive-mount cell reusing the reference pattern (`drive.mount('/content/drive')`), plus a check that `/content/drive/MyDrive/SmartPlug` exists and contains `plug_dump_*.csv` files, raising an actionable Russian error message with the expected path otherwise. Verify: running with a missing folder shows the clear error; `drive.mount` call matches the reference notebook.
- [x] 3.2 Implement the ingestion function with a docstring and Russian comments, reusing the reference notebook logic verbatim: `GetLabelDuration` regex extraction of label and duration from `plug_dump_...csv` filenames; read each CSV with `;` delimiter; extract ` P_RMS, mW`; cut into consecutive 90-point chunks; skip files with unparseable names or <90 samples with a visible warning; exclude labels containing `IdleCharge`/`Browsing`/`VKAudio`; return the (N, 90) feature matrix and parallel label list; print chunk total and per-class distribution. Verify: docstring present; logic textually matches the reference cells' parameters (regex, delimiter, column name, CHUNK_LENGTH=90, excluded substrings); a Colab run against the user's Drive prints a non-empty distribution.
- [x] 3.3 Add a small cell printing dataset shape, per-class counts exactly as chunks, and a warning when rows exceed the ~5000-row CPU limit (with the stratified-subsampling note for CPU-only runs per design.md). Verify: prints reflect the ingested matrix; the CPU-limit warning text is present in the cell.

## 4. Zero-shot evaluation with TabPFN-3.5

- [x] 4.1 Add section markdown cell ("Классификация сценариев TabPFN-3.5"), then an evaluation cell with commented steps and timing: stratified `train_test_split(test_size=0.3, random_state=42)`; `TabPFNClassifier` (default `ModelVersion.V3_5`); `fit(X_train, y_train)` with elapsed time; single `predict(X_test)` call with elapsed time; no scaling or one-hot anywhere. Verify: no `StandardScaler`/`OneHotEncoder` usage in the notebook; seed fixed; timings printed.
- [x] 4.2 Add a metrics cell printing accuracy, `classification_report`, and the confusion matrix rendered via `ConfusionMatrixDisplay` (matplotlib, no new dependency). Verify: on a Colab run all three outputs appear; plot uses only preinstalled imports.

## 5. Variant comparison with TabPFN-3.5-Fast

- [x] 5.1 Add section markdown cell ("Сравнение со TabPFN-3.5-Fast"), then a cell repeating task 4.1–4.2's protocol with `ModelVersion.V3_5_FAST` and storing its accuracy/timings next to the default variant's, ending in a printed side-by-side summary (accuracy, fit seconds, predict seconds). Verify: both variants appear in the summary output; no duplicated import blocks or dead code between the two evaluation cells.

## 6. Conclusions

- [x] 6.1 Add a final markdown cell in Russian answering the user's two questions from both sources: what the docs say (resource limits from task 2.4) and what the run showed (timings, GPU/CPU facts, T4 fit), plus a practical recommendation (GPU recommended for speed; T4 more than sufficient; TabPFN-3.5-Fast as the fast option; non-commercial license reminder). No code in this cell.

## 7. Hygiene self-review and integration check

- [x] 7.1 Perform the notebook-wide self-review pass: remove unused imports, dead code, leftover debug cells; confirm every code cell starts with a "what & why" comment, every defined function/class has a docstring, and every logical block is preceded by a section markdown cell. Verify: an automated source scan of the notebook reports zero unused imports and every `def`/`class` line is followed by a docstring.
- [x] 7.2 Full integration dry-run: check notebook JSON validity, cell ordering, and that a cached `tabpfn` install makes the run reproducible end-to-end on a fresh Colab session (Run All from an empty runtime, T4). Verify: no errors in cell outputs; final metrics cell is the last executed cell.

## 8. License access for model weights (added during apply)

Discovered during the first Colab run: `tabpfn` gates the weight download behind a one-time license acceptance (Prior Labs API key in `TABPFN_TOKEN`; Colab has no interactive terminal, so the browser flow is unusable). Runtime error: `TabPFNLicenseError` at first `fit`.

- [x] 8.1 Add a license section to the notebook (markdown with the one-time registration/acceptance/key instruction for ux.priorlabs.ai, then a code cell that sources `TABPFN_TOKEN` from the environment, Colab Secrets (`userdata`), or an in-cell variable and sets it before `fit`), renumbering the pipeline sections; Verify: static scanner reports the `TABPFN_TOKEN`/ux.priorlabs.ai fragments and sections 1..8, zero unused imports, all docstrings present.
- [x] 8.2 Update planning artifacts for the revealed change: add the license-handling requirement to the spec delta, correct the design risk that claimed no token was needed, and record the Prior Labs account prerequisite in the proposal; Verify: `openspec validate --changes ...` passes.
- [x] 8.3 Re-run all local verification (static scanner + ingestion smoke test) on the rebuilt notebook; Verify: both scripts pass end-to-end. Colab-side confirmation of the full run remains with the user (task 7.2 flow).

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Verify the archived result.
