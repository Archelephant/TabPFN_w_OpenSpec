# Design

## Context

This change adds a new analysis notebook; the only existing artifact is `TimesFM3_DeepSeek.ipynb`, which already defines the ingestion pattern the new notebook must follow. Relevant facts established during planning:

- **Data** (`TimesFM3_DeepSeek.ipynb`): Google Drive folder `/content/drive/MyDrive/SmartPlug` holds dump files named `plug_dump_<yyyy_mm_dd_hh_mm_ss>_<label>_<duration>.csv` (`;`-separated). Each file is a continuous power series; the reference pipeline extracts label/duration by regex, reads column ` P_RMS, mW`, cuts it into 90-point chunks, tags each chunk with the filename label, and skips label substrings `IdleCharge`/`Browsing`/`VKAudio` (also `MixedBrowsing`, subsumed by `Browsing`). Result in the reference runs: a feature matrix (n, 90) plus a label list over several appliance/usage classes.
- **TabPFN-3.5** (arXiv 2609.17895; HF `Prior-Labs/tabpfn_3_5`; repo PriorLabs/TabPFN): ~220M-parameter in-context tabular transformer, one checkpoint serves classification and regression; PyPI package `tabpfn`, Python ≥3.10; checkpoint auto-downloads from Hugging Face on first `fit`. Documented resource guidance: GPU recommended (~8 GB VRAM works; 16 GB only for some large datasets); on CPU, moderate datasets (≤ ~5000 rows) are feasible. Usage guidance: do **not** scale or one-hot encode inputs; batch test rows (< ~1000 per predict call); TabPFN-3.5 accepts up to 20k features / ~1M rows on GPU. Weights are non-commercial licensed but explicitly allow testing/evaluation.
- **Colab free tier**: T4 GPU runtime = 16 GB VRAM + 12.7 GB RAM; CPU runtime has no VRAM. Colab preinstalls torch, pandas, numpy, scikit-learn, tqdm, matplotlib.

The user's two resource questions are hereby answered at plan level: a GPU is **recommended but not mandatory** for this dataset (few thousand rows × 90 features), and **even the T4 is far more than needed** (220M parameters, fp32 weights ≈ <1 GB; the 16 GB figure concerns very large contexts).

## Goals / Non-Goals

**Goals:**

- One notebook that runs unmodified in Colab (T4 or CPU runtime) and empirically demonstrates TabPFN-3.5 zero-shot scenario classification on the SmartPlug traces, printing GPU/CPU facts and timings so the resource questions are answered at runtime too.
- Honest quality bar for the deliverable: commented cells, docstrings, markdown sections, zero unused imports (explicit user requirement).

**Non-Goals:**

- No fine-tuning / no classification heads on top of TabPFN (the TimesFM3 notebook's "open the model and add layers" approaches are not repeated; TabPFN's API does not expose or need them).
- No forecasting, no TimesFM integration, no comparison against timesfm/RandomForest results (a possible follow-up change).
- No regression evaluation with the same checkpoint (assumption recorded in the proposal).
- No changes to existing project files; no local-runnable validation on this machine (data lives in the user's Google Drive; Colab is the only test environment).

## Decisions

- **Deliverable is one notebook at project root** (`TabPFN35_SmartPlug.ipynb`), matching the existing project convention of self-contained top-level Colab notebooks (and the Google-Colab notebook metadata format, as in the reference notebook). Alternative: `notebooks/` subfolder — rejected; a single project with one notebook does not need it.

- **Use TabPFN's local inference package, not the hosted client.** The goal is to test *the model* under Colab resource constraints; the cloud `tabpfn-client` would test someone's service. `pip install tabpfn` is the only install; everything else is Colab-preinstalled. Alternative: tabpfn-client — rejected because it bypasses the GPU/CPU questions this change exists to answer.

- **Frame the task as plain tabular classification: one 90-point chunk = one row of 90 features.** This reuses the reference notebook's ingestion verbatim and plays to TabPFN-3.5's claim (raw wide tabular data, no preprocessing). Alternative: reuse the reference notebook's TimesFM feature engineering first — rejected: it would test the pipeline, not the model, and doubles the dependency burden (timesfm install), violating the minimize-dependencies constraint.

- **Evaluation protocol: stratified 70/30 split, seed 42, then `fit`/`predict` on the raw matrix** — identical to the reference notebook's protocol so results are comparable. Accuracy + classification report + confusion matrix (printed via `ConfusionMatrixDisplay` with matplotlib — preinstalled, no new dependency) + `time.time()` timings around fit and predict. Predict called once on the whole test set (test set << 1000 rows makes the batching rule moot; if it ever exceeded 1000 rows, chunk predict calls).

- **Evaluate TabPFN-3.5-Fast as the second variant** (`ModelVersion.V3_5_FAST`). One code cell, no new imports, and it directly serves the user's speed/feasibility concern; results are reported side-by-side in a final markdown-free metrics summary cell. Alternative: default variant only — rejected as throwing away a free comparison.

- **Explicit environment-report cell early in the notebook**: prints `torch.__version__`, CUDA availability, GPU name and VRAM (`torch.cuda.get_device_properties`), plus the documented resource facts restated in the adjacent markdown cell and a printed verdict line (GPU found / running on CPU). This turns the user's two questions into observable notebook output, not just planning prose.

- **Notebook authoring style** (explicit user requirements): Russian text/comments consistent with `TimesFM3_DeepSeek.ipynb`; a markdown introduction cell before each logical section; a detailed comment (what/why) atop every code cell; docstrings on every defined function/class; a final self-review pass to delete dead code and unused imports.

- **Robust ingestion**: reuse the reference notebook's regex labels/duration extraction and chunk logic unchanged, wrapped so that a missing SmartPlug folder raises an actionable error and unreadable/short files are skipped with a visible warning rather than failing the run.

## Risks / Trade-offs

- [Dataset size over TabPFN's CPU limit if run without GPU and the user keeps all classes] → The notebook prints the row count; if rows exceed ~5000 on a CPU-only runtime it warns and (documented fallback) stratified-subsamples to the limit; on T4 the full set is used.
- [Class imbalance after label filtering] → stratified split is mandatory (already in the protocol); confusion matrix and per-class report make imbalance visible.
- [Checkpoint download must pass the Prior Labs license gate] → The `tabpfn` package requires a one-time license acceptance before downloading weights; in Colab (no interactive terminal) the browser flow is unavailable, so the notebook configures `TABPFN_TOKEN` from the environment, Colab Secrets, or an in-cell variable and prints the full registration/acceptance/key instructions when no token is present (free account at ux.priorlabs.ai). Network failure thereafter produces a clear HF error — documented in the notebook's intro cell.
- [Colab Python/toolchain drift breaking `tabpfn` install] → `pip install tabpfn` resolves a compatible version; no version pinning planned (minimizes maintenance); the notebook is re-runnable and fails loudly rather than silently.
- [Exact checkpoint file size on HF unknown at planning time] → runtime facts section intentionally prints the downloaded checkpoint info; not blocking: even a multi-hundred-MB artifact fits Colab disk/RAM/T4 easily for a 220M-param model.
- [Wall-clock performance on CPU can be slow to the point of feeling broken] → timings are printed next to results so the CPU cost is visible; TabPFN-3.5-Fast is the recommended escape hatch in the notebook's conclusion cell.

## Migration Plan

Single new file; nothing to migrate. Rollback = delete `TabPFN35_SmartPlug.ipynb`. Verification is by Colab smoke run (T4 and CPU runtimes) and the notebook-hygiene review from the spec.

## Open Questions

- Exact label set/count and chunk totals the user's current SmartPlug folder yields — depends on data not available on this machine; only affects printed results, not the design.
