# Extract-Then-Accelerate

This repository contains the code, trained model artifacts, and experiment outputs for the 2026 IISE submission.

## Repository layout

- `ai4seru/`: heuristic and metaheuristic algorithms, utilities, and tests.
- `model/`: graph neural network models, configuration files, trained weights, and XAI utilities.
- `experiment/`: experimental results, tables, and figures.

## Environment

The CUDA/GNN environment used locally is `gnn_cuda12.8`:

- Python: 3.9.23
- CUDA runtime reported by PyTorch: 12.8
- PyTorch: 2.8.0+cu128
- CUDA availability in the local environment: `True`

Core packages from `gnn_cuda12.8` that are used by this project:

| Package | Version | Purpose |
| --- | --- | --- |
| `torch` | 2.8.0+cu128 | GNN model training and inference |
| `torch-geometric` | 2.6.1 | Graph data structures and loaders |
| `torch-scatter` | 2.1.2+pt28cu128 | PyG CUDA extension |
| `torch-sparse` | 0.6.18+pt28cu128 | PyG sparse CUDA extension |
| `torch-cluster` | 1.6.3+pt28cu128 | PyG clustering CUDA extension |
| `torch-spline-conv` | 1.2.2+pt28cu128 | PyG spline convolution extension |
| `torchaudio` | 2.8.0.dev20250825+cu128 | Installed with the CUDA PyTorch stack |
| `torchvision` | 0.24.0.dev20250825+cu128 | Installed with the CUDA PyTorch stack |
| `cuda-toolkit` | 12.8.1 | CUDA toolkit |
| `cuda-compiler` | 12.8.1 | CUDA compiler tools |
| `cuda-libraries` | 12.8.1 | CUDA runtime libraries |
| `cuda-version` | 12.8 | CUDA version metapackage |
| `cudnn` | 9.13.0.50 | cuDNN runtime |
| `libcudnn` | 9.13.0.50 | cuDNN library |
| `numpy` | 1.26.4 | Numerical computation |
| `pandas` | 1.5.3 | Tables, experiment outputs, and CSV/XLSX processing |
| `scipy` | 1.13.1 | Statistical tests and signal processing |
| `scikit-learn` | 1.6.1 | Metrics, clustering, splits, and surrogate modeling utilities |
| `matplotlib` | 3.9.4 | Figures and convergence plots |
| `openpyxl` | 3.1.5 | Excel `.xlsx` I/O |
| `xlrd` | 2.0.2 | Excel `.xls` I/O |
| `pyyaml` | 6.0.3 | YAML configuration files |
| `requests` | 2.32.5 | GNN API calls |
| `tqdm` | 4.67.1 | Progress bars |
| `networkx` | 3.2.1 | Graph-related utilities |
| `pillow` | 11.3.0 | Image backend used by plotting tools |
| `pytest` | 8.4.2 | Test runner |

Some scripts import additional optional packages that were not present in the captured `gnn_cuda12.8` package list:

- `flask`: API server scripts in `model/` and `ai4seru/metaheuristics/ccea/ccea_java.py`.
- `deap`: genetic algorithm operators in `ai4seru/metaheuristics/common/`.
- `xgboost` and `shap`: XAI surrogate modeling and SHAP plots in `model/util/xai_*`.
- `scikit-posthocs`: post-hoc statistical tests in `ai4seru/test_gnn_api.py`.
- `gymnasium` and `stable-baselines3`: archived RL feature extractor code under `ai4seru/z_trash/`.

The local environment can be activated with:

```bash
conda activate gnn_cuda12.8
```

## Large files

Large data, model, and result files are tracked with Git LFS. After cloning the repository, install Git LFS and pull the large files:

```bash
git lfs install
git lfs pull
```
