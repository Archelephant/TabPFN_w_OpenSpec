# Proposal

## Why

A new state-of-the-art tabular foundation model, TabPFN-3.5 (arXiv 2609.17895, Sep 2026; weights on Hugging Face `Prior-Labs/tabpfn_3_5`), is available and claims large gains over TabPFN-3 with zero-shot inference (no hyperparameter tuning, no per-dataset preprocessing). We have no verification yet of how it performs on our user energy-consumption scenario data (SmartPlug appliance power traces), and no verified answer to the user's resource questions: does TabPFN-3.5 need a GPU, and does it fit in the free Google Colab T4? A single, runnable Colab notebook answers both.

## What Changes

- Add a new Colab-compatible notebook `TabPFN35_SmartPlug.ipynb` at the project root:
  - Install only the `tabpfn` pip package (all other imports come pre-installed in Colab: `pandas`, `numpy`, `scikit-learn`, `torch`).
  - Mount Google Drive and load the SmartPlug data exactly as in `TimesFM3_DeepSeek.ipynb`: discover `plug_dump_<timestamp>_<label>_<duration>.csv` files in `/content/drive/MyDrive/SmartPlug`, parse the scenario label from the filename (`;` delimiter), and cut the ` P_RMS, mW` power series into 90-point rows.
  - Detect and report the runtime environment (CUDA available / GPU name / VRAM) and print the documented TabPFN-3.5 resource requirements so the GPU-vs-CPU and T4-fit questions are answered empirically, not just from docs.
  - Fit and evaluate the default TabPFN-3.5 classifier (`TabPFNClassifier` with `ModelVersion.V3_5`) on the scenario-label classification task: stratified train/test split, accuracy, classification report, confusion matrix; record wall-clock time for fit/predict.
  - Repeat the evaluation with the smaller TabPFN-3.5-Fast variant for a runtime/accuracy comparison.
- Follow the requested quality rules for the notebook: every cell commented in detail (what and why), docstrings for every function/class, markdown section cells between logical blocks, no unused imports or leftover code.
- No changes to the existing `TimesFM3_DeepSeek.ipynb` or any other project file.

Recorded assumptions (made during planning, verify in review if material):
- Task is **classification** of the scenario label from a 90-point power trace, mirroring the reference notebook; the regression head of TabPFN-3.5 is not exercised.
- The noisy-label filter (exclude `IdleCharge`, `Browsing`, `MixedBrowsing`, `VKAudio` label substrings) is kept exactly as in the reference notebook.
- The SmartPlug CSV folder already exists in the user's Google Drive under `/content/drive/MyDrive/SmartPlug` (the notebook mounts GDrive like the reference notebook does and fails clearly if the folder is missing).
- Notebook text and comments are written in Russian, consistent with `TimesFM3_DeepSeek.ipynb`.
- The user has (or creates) a free Prior Labs account at ux.priorlabs.ai: the `tabpfn` package requires one-time license acceptance and a `TABPFN_TOKEN` API key before it downloads the TabPFN-3.5 checkpoint (discovered during the first Colab run; Colab has no interactive terminal for the browser flow).

## Capabilities

### New Capabilities

- `smartplug-scenario-classification`: a reproducible Google Colab notebook pipeline that turns SmartPlug power-trace CSV dumps into a labeled tabular dataset (90-point power rows, filename-borne scenario labels) and evaluates a tabular foundation model with zero-shot fit/predict on the scenario-classification task, including environment/resource checks and runtime metrics.

### Modified Capabilities

- (none — the project has no existing specs)

## Impact

- **New file**: `TabPFN35_SmartPlug.ipynb` at the project root. Nothing else is created, modified, or removed.
- **Dependencies (Colab runtime only)**: `pip install tabpfn` (PyPI). The 220M-parameter TabPFN-3.5 checkpoint is downloaded from Hugging Face on first `fit` — after a one-time license acceptance (free Prior Labs account at ux.priorlabs.ai; API key passed via `TABPFN_TOKEN`). All other runtime dependencies (pandas, numpy, scikit-learn, torch, tqdm, matplotlib/seaborn for plots) are already present in Colab. No project-level dependency changes.
- **Hardware**: analysis of the technical report and the repository README shows TabPFN-3.5 (220M parameters, ~1M-row context limit) needs ~8 GB VRAM for typical datasets, 16 GB for very large ones; CPU inference is feasible for moderate datasets (up to ~5000 rows). The SmartPlug dataset (90 features, a few thousand rows) therefore fits a Colab T4 (16 GB VRAM) with room to spare; the notebook must also verify this empirically at runtime.
- **Data**: read-only access to the user's Google Drive `SmartPlug` folder; no data is written back.
- **License**: model weights are non-commercial (testing/evaluation explicitly allowed) — in scope for this evaluation use case.
