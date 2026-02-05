## Python environment
python == 3.10.12

## Required Paths (Used in All Three Stages)

### `--csv_dir`
Directory containing CSV files for dataset splits of the target mutation task.

- Each CSV corresponds to one fold of the mutation-specific dataset
- If this directory is empty, 4-fold CSV files will be automatically generated.
The generation requires `clinical_pkl_path` (mutation labels), and the resulting
CSV files will be saved under this directory.

---

### `--feature_dir`
Directory containing extracted feature files (`.pt`).

- Features are typically extracted using a pretrained backbone (e.g., CHIEF)

---

### `--h5_dir`
Directory containing HDF5 coordinates files (`.h5`).

- Contains coordinates
- The directory must contain feature files whose names correspond to the slide
identifiers (e.g., `TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01`).

**Feature and h5 dir example:**
```
TCGA-LUAD-FS/
└── CHIEF/
    └── 20X/
        ├── h5_files(stain_norm)/
        │   ├── TCGA-XX-XXXX-01A-01-TS1.h5
        │   ├── TCGA-XX-XXXX-01A-01-TS2.h5
        │   └── ...
        │
        ├── pt_files(stain_norm)/
        │   ├── TCGA-XX-XXXX-01A-01-TS1.pt
        │   ├── TCGA-XX-XXXX-01A-01-TS2.pt
        │   └── ...
        │
        └── cluster_record_spatialleiden.pkl
```

---

### `--clinical_pkl_path`
Path to the clinical information pickle file containing labels for the target mutation task.

**Clinical pickle example:**

| case_submitter_id | mutation |
|------------------|----------|
| TCGA-05-4249     | 0        |
| TCGA-05-4382     | 0        |
| TCGA-05-4389     | 0        |

---

### `--cluster_pkl_dir`
Directory to the clustering result pickle file.

- Stores precomputed cluster assignments of patch-level features
- Used to construct group-level representations or constraints
- The same `cluster.pkl` can be shared across different mutation tasks
  as long as the tumor type and the underlying feature extraction backbone
  remain the same

---

### `--save_dir`
Directory for saving outputs and checkpoints.

- Training logs, model pth, and evaluation results are saved here
- A separate subdirectory is created for each fold and stage

**Save structure: (after saving)**

```
save-dir/
    ├── data
    │   ├── cluster.pkl
    │   ├── dataset_fold_0.csv
    │   ├── dataset_fold_1.csv
    │   ├── dataset_fold_2.csv
    │   └── dataset_fold_3.csv
    │
    ├── groupConstraint 
    │   ├── dataset_fold_0
    │   │   ├── abmil.pth
    |   |   ├── group_abmil.pth
    |   |   ├── probability.csv
    │   │   └── static.csv
    │   │
    │   ├── dataset_fold_1
    │   ├── dataset_fold_2
    │   └── dataset_fold_3
    │
    ├── baseline
    │   ├── dataset_fold_0
    │   │   ├── basemodel.pth
    |   |   ├── probability.csv
    │   │   └── static.csv
    │   │
    │   ├── dataset_fold_1
    │   ├── dataset_fold_2
    │   └── dataset_fold_3
    │
    ├── fold_results.csv
    └── summary_pValuse_results.csv

```

---

### `--config`
Path to the YAML configuration file.

- Contains hyperparameters shared across all stages
- Allows reproducible experiment configuration

## Optional paths (In training stage)
### `--test_dir`

Path to a directory containing trained GroupConstraintMIL results.

- Required when `--train` is `false` and `--eval_static` is `true`
- Please provide the **parent directory** of the `groupConstraint` folder
- The pipeline will load trained models and evaluation results from this directory, ex: test-root.
- Missing or incorrect paths may result in failed evaluation

**groupConstraint dir structure: (after saving)**

```
test-root/
    └── groupConstraint 
        ├── dataset_fold_0
        │   ├── abmil.pth
        |   ├── group_abmil.pth
        |   ├── probability.csv
        │   └── static.csv
        │
        ├── dataset_fold_1
        ├── dataset_fold_2
        └── dataset_fold_3
```

### `--baseline_dir`
Path to a directory containing trained baseline results.

- Please provide the **parent directory** of the `baseline` folder
- The pipeline will load trained models and evaluation results from this directory, ex: test-root.
- If the specified path is missing or incorrect, baseline training will be
  automatically triggered and saved under `save_dir/baseline`

**baseline dir structure: (after saving)**

```
test-root/
    └── baseline 
        ├── dataset_fold_0
        │   ├── basemodel.pth
        |   ├── probability.csv
        │   └── static.csv
        │
        ├── dataset_fold_1
        ├── dataset_fold_2
        └── dataset_fold_3
```

## Configuration Usage

The pipeline supports running with a YAML configuration file only.

You may launch the pipeline using:

```bash
python main.py \
  --config "yaml path"
```

### Configuration Priority

When a YAML configuration file is provided, only parameters explicitly defined
in the YAML file will **override** their corresponding command-line arguments
(`--parameters`).

> Priority rule: YAML configuration > command-line arguments


## Execution Order

The pipeline can be executed in two stages, but it also supports running the
training stage directly.

### Stage 1: Preprocessing (Optional)
The following scripts can be executed **simultaneously**:

- `run_clustering.sh`
- `run_train_baseline.sh`

This stage prepares clustering information and baseline results used for
subsequent training.

### Stage 2: Training
- `run_train.sh`

---

### Flexible Execution

You may choose one of the following execution modes:

1. **Two-stage execution**  
   First run clustering and baseline scripts, then run training:
   ```text
   run_clustering.sh + run_train_baseline.sh  →  run_train.sh
2. **Direct training execution (recommended)** 

    Run `run_train.sh` directly without running the preprocessing stage.

    - If clustering results are not found, they will be generated automatically
    - If baseline results are missing, the baseline training will be executed automatically


## Training parameters

### `--group_lr`
group level lr
### `--lr`
Patch level lr 
### `--theta_start` and `--theta_end`
The theta value is constrained to lie within this range.
It is recommended that both values are ≥ 1, and that `theta_start` > `theta_end`.
### `--k`
Controls the slope of the sigmoid function about theta_function.
**Formula**
\[
\theta = \theta_{\text{start}}
+ (\theta_{\text{end}} - \theta_{\text{start}})
\cdot \sigma\big(
k \cdot (\text{group\_attention}
- \overline{\text{group\_attention}})
\big)
\]
### `--dirichlet_weight`
Controls the strength of the Dirichlet regularization.

