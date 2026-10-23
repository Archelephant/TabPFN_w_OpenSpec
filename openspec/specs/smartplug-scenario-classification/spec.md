# smartplug-scenario-classification Specification

## Purpose

Provide a reproducible Google Colab notebook that verifies TabPFN-3.5 (the new tabular foundation model) on user energy-consumption scenario data by classifying SmartPlug power-trace chunks, with empirical answers to whether a GPU is required and whether the free Colab T4 is sufficient, using a minimal dependency footprint.

## Requirements

### Requirement: Colab-runnable evaluation notebook

The change SHALL deliver a single `.ipynb` notebook that, after the user connects the runtime and provides the SmartPlug Google Drive folder, executes end-to-end (Run All) in Google Colab without editing any cell, on both a T4 GPU runtime and a CPU-only runtime.

#### Scenario: Full run on T4 runtime

- **WHEN** the notebook is run top-to-bottom in Colab with a T4 GPU runtime and a mounted Google Drive containing the SmartPlug CSV dumps
- **THEN** every cell completes without errors, and final metric cells print evaluation results

#### Scenario: Fall back to CPU-only runtime

- **WHEN** the notebook is run top-to-bottom in Colab with a CPU-only runtime (and the dataset row count is within TabPFN's CPU-profile limit of ~5000 rows)
- **THEN** the notebook completes without requesting a GPU, and reports the environment used in the results

### Requirement: Minimal dependency footprint

The notebook SHALL install only the `tabpfn` package via pip and SHALL NOT pip-install any other package; every import in the notebook SHALL come either from `tabpfn`, or from packages pre-installed in the Colab environment (e.g. pandas, numpy, scikit-learn, torch, tqdm), or from Colab built-in magics.

#### Scenario: Dependency installation cell

- **WHEN** the installation cell is executed
- **THEN** `pip install tabpfn` is the only non-builtin package installation performed

#### Scenario: No stray dependencies after review

- **WHEN** the generated notebook is reviewed for imports
- **THEN** no import corresponds to a package that requires manual installation in Colab other than `tabpfn`, and no unused import or dead code block remains

### Requirement: SmartPlug trace discovery and label parsing

The pipeline SHALL mount Google Drive, discover files matching `plug_dump_<yyyy-mm-dd hh-mm-ss>_<label>_<duration>.csv` in the SmartPlug folder, extract the scenario label and duration from each filename, and skip any file whose name does not match the pattern.

#### Scenario: Successful ingestion

- **WHEN** the SmartPlug folder exists in the mounted Google Drive and contains at least one potentially valid `plug_dump_*.csv` file
- **THEN** ingestion completes with one row per 90-point chunk whose label equals the filename label

#### Scenario: Missing data folder

- **WHEN** the SmartPlug folder is absent or empty in the mounted Google Drive
- **THEN** ingestion stops with an explicit, actionable error message identifying the expected folder path, rather than producing an empty dataset

#### Scenario: Malformed filenames

- **WHEN** a file does not match the filename pattern (label/duration not extractable)
- **THEN** the file is skipped, the skip is either warned about or logged visibly, and ingestion continues with the remaining files

### Requirement: Power series to labeled tabular rows

The pipeline SHALL read each discovered CSV with the `;` delimiter, extract the ` P_RMS, mW` power column, cut each series into consecutive 90-point rows, and exclude labels containing `IdleCharge`, `Browsing`, or `VKAudio` substrings as in the reference notebook, producing a feature matrix and a parallel label list and reporting chunk counts per class.

#### Scenario: Chunking a valid dump file

- **WHEN** a discovered CSV contains ≥ 90 power samples
- **THEN** each run of 90 consecutive samples becomes exactly one row labeled with the file's scenario label, and the notebook prints the per-class chunk distribution

#### Scenario: Too-short file

- **WHEN** a discovered CSV contains fewer than 90 samples
- **THEN** the file contributes no rows, the skip is visible in the output, and ingestion continues

### Requirement: Runtime environment report

The notebook SHALL print whether CUDA is available, the GPU device name and total VRAM when a GPU is present, and SHALL restate TabPFN-3.5's documented resource characteristics (GPU recommended ~8 GB VRAM, up to 16 GB needed only for very large datasets, CPU feasible for ~5000 rows), so that the user questions "GPU required or CPU is enough?" and "does it fit a Colab T4?" are answered both from documentation and from the actual observed runtime.

#### Scenario: GPU runtime report

- **WHEN** the environment-report cell runs on a T4 runtime
- **THEN** the output states CUDA availability, identifies the GPU as Tesla T4, and shows its 16 GB VRAM

#### Scenario: CPU-only runtime report

- **WHEN** the environment-report cell runs on a CPU-only runtime
- **THEN** the output states that CUDA is unavailable and that the evaluation proceeds on CPU, citing the documented row-count limit under which TabPFN-3.5 remains feasible on CPU

### Requirement: Zero-shot scenario classification evaluation

The pipeline SHALL split the ingested dataset into stratified train/test partitions (30% test, fixed random seed), fit TabPFN-3.5 (`TabPFNClassifier` with the default `ModelVersion.V3_5`) on the training rows, predict the test rows with no input scaling or one-hot encoding beyond what was ingested, and print accuracy, a per-class classification report, a confusion matrix, and elapsed fit/predict times.

#### Scenario: Evaluation on ingested dataset

- **WHEN** ingestion has produced a feature matrix and labels
- **THEN** fit and predict complete without dataset-specific hyperparameter tuning, and the reported metrics include accuracy, classification report, confusion matrix, and timings

#### Scenario: Reproducible split

- **WHEN** the evaluation is run twice with the same fixed seed
- **THEN** the train/test partition is identical in both runs

### Requirement: Variant comparison with TabPFN-3.5-Fast

The notebook SHALL repeat the same evaluation using TabPFN-3.5-Fast (`ModelVersion.V3_5_FAST`) and present its accuracy and timings next to the default TabPFN-3.5 results for a runtime/accuracy comparison.

#### Scenario: Both variants reported

- **WHEN** the evaluation cells have run for both variants
- **THEN** the notebook output contains metric summaries for TabPFN-3.5 and TabPFN-3.5-Fast with durations and accuracy that can be compared directly

### Requirement: License acceptance for model weights

The notebook SHALL support TabPFN's license-gated weight download: before the first fit it SHALL source `TABPFN_TOKEN` from the environment variable, Google Colab Secrets, or an in-cell variable, set the environment variable when a token is found, and otherwise print one-time registration/acceptance instructions (ux.priorlabs.ai) in Russian instead of failing at fit.

#### Scenario: Token provided

- **WHEN** `TABPFN_TOKEN` is available from the environment, Colab Secrets, or the in-cell variable
- **THEN** the notebook sets the `TABPFN_TOKEN` environment variable before the evaluation, and the first `fit` proceeds to download the checkpoint without a `TabPFNLicenseError`

#### Scenario: Token absent

- **WHEN** no supported token source has a value
- **THEN** the license cell prints the full one-time instructions (register/use ux.priorlabs.ai, accept the license on the Licenses tab, copy the API key from the account page, put it into `TABPFN_TOKEN`) and warns that the first `fit` will end with `TabPFNLicenseError` otherwise

### Requirement: Notebook documentation hygiene

The notebook SHALL divide the pipeline into logical sections separated by markdown cells, every code cell SHALL carry explanatory comments stating what happens and why, and every function or class defined in the notebook SHALL have a docstring.

#### Scenario: Hygiene review

- **WHEN** the generated notebook is reviewed cell-by-cell
- **THEN** each section is introduced by a markdown cell, each code cell has explanatory comments, each defined function or class has a docstring, and no cell contains dead or unused code
