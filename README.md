# mlr3learners.gpytorch

[![R-CMD-check](https://github.com/KhalifaSeck/mlr3learners.gpytorch/actions/workflows/check.yaml/badge.svg)](https://github.com/KhalifaSeck/mlr3learners.gpytorch/actions/workflows/check.yaml)
[![pkgdown](https://github.com/KhalifaSeck/mlr3learners.gpytorch/actions/workflows/pkgdown.yaml/badge.svg)](https://github.com/KhalifaSeck/mlr3learners.gpytorch/actions/workflows/pkgdown.yaml)
[![Codecov test coverage](https://codecov.io/gh/KhalifaSeck/mlr3learners.gpytorch/branch/master/graph/badge.svg)](https://app.codecov.io/gh/KhalifaSeck/mlr3learners.gpytorch?branch=master)
[![Netlify Status](https://api.netlify.com/api/v1/badges/VOTRE-SITE-ID/deploy-status)](https://mlr3learners-gpytorch.netlify.app)

Gaussian Process regression learner for mlr3 using GPyTorch with GPU acceleration support.

**Documentation**: https://mlr3learners-gpytorch.netlify.app

## Installation

### Complete Installation (Recommended)

```r
# Step 1: Install R package
remotes::install_github("KhalifaSeck/mlr3learners.gpytorch")

# Step 2: Install miniconda (one-time only)
if(!reticulate::miniconda_exists()) {
  reticulate::install_miniconda()
}

# Step 3: Create Python environment (one-time only)
if(!reticulate::conda_exists("gpytorch")) {
  reticulate::conda_create("gpytorch")
  reticulate::conda_install("gpytorch", c("pytorch", "gpytorch"))
}

# Step 4: Configure Python path
Sys.setenv(RETICULATE_PYTHON="~/AppData/Local/r-miniconda/envs/gpytorch/python.exe")  # Windows
# Sys.setenv(RETICULATE_PYTHON="~/miniconda3/envs/gpytorch/bin/python")  # Linux/Mac

# Step 5: Activate environment
reticulate::use_condaenv("gpytorch", required = TRUE)
```

### Verify Installation

```r
library(mlr3)
library(mlr3learners.gpytorch)

task <- mlr3::tsk("mtcars")
learner <- mlr3::lrn("regr.gpytorch", n_iter = 50)
learner$train(task)
pred <- learner$predict(task)
print(pred)
```

### Manual Python Installation

If you prefer managing Python separately:

**CPU-only:**
```bash
conda create -n gpytorch python=3.10
conda activate gpytorch
conda install pytorch -c pytorch
pip install gpytorch
```

**GPU (NVIDIA CUDA):**
```bash
conda create -n gpytorch python=3.10
conda activate gpytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gpytorch
```

Then in R:
```r
reticulate::use_condaenv("gpytorch", required = TRUE)
```

## Usage

### Basic Example

```r
library(mlr3)
library(mlr3learners.gpytorch)
library(reticulate)

use_condaenv("gpytorch")

task <- tsk("mtcars")
learner <- lrn("regr.gpytorch", n_iter = 50)

learner$train(task)
pred <- learner$predict(task)
print(pred)
```

### GPU Usage

```r
learner_gpu <- lrn("regr.gpytorch", n_iter = 50, device = "cuda")
learner_cpu <- lrn("regr.gpytorch", n_iter = 50, device = "cpu")
learner_auto <- lrn("regr.gpytorch", n_iter = 50, device = "auto")
```

## Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| kernel | factor | "rbf" | {rbf, matern} | Kernel type for covariance function |
| lr | numeric | 0.1 | [0.001, 1] | Learning rate for Adam optimizer |
| n_iter | integer | 50 | [10, 500] | Number of training iterations |
| device | factor | "auto" | {auto, cpu, cuda} | Computing device (auto detects GPU) |

## Benchmark Results

Performance evaluated using Mean Squared Error (MSE) in 5-fold cross-validation.

### Overall Performance

| Dataset | GPyTorch | KNN | CV-Glmnet | Featureless |
|---------|----------|-----|-----------|-------------|
| airquality | 398.58 | 388.28 | 575.86 | 1107.45 |
| boston | **9.68** | 18.02 | 27.79 | 84.70 |

Lower MSE indicates better performance. GPyTorch ranks #1 on boston dataset.

### Analysis

**Does GPyTorch learn non-trivial patterns?**

- airquality: 64% improvement over featureless baseline
- boston: 89% improvement over featureless baseline

**Is GPyTorch competitive with other algorithms?**

- Ranks #1 on boston dataset
- Ranks #2 on airquality dataset
- Outperforms linear models on both datasets

## GPU vs CPU Performance

Performance comparison on NVIDIA GeForce RTX 3050.

### Prediction Accuracy

GPU and CPU produce identical predictions:
- airquality: 398.58 (GPU) = 398.58 (CPU)
- boston: 9.68 (GPU) = 9.68 (CPU)

### Training Time (30 iterations)

**Small dataset (111 obs):** CPU 2.5x faster (GPU overhead dominates)  
**Medium dataset (506 obs):** GPU 1.3x faster (computation outweighs overhead)  
**Expected for large datasets (1000+ obs):** GPU 2-10x faster

**Recommendation:** Use CPU for small datasets, GPU for large datasets.

## Technical Features

- **GPU/CPU support**: Automatic device detection or manual selection
- **Feature normalization**: Automatic zero mean, unit variance scaling
- **Kernel options**: RBF and Matérn kernels
- **Numerical stability**: Proper variance handling and scaling

## System Requirements

**CPU-only:** Modern multi-core CPU, 4GB RAM  
**GPU:** NVIDIA GPU with CUDA 11.8+, 4GB GPU memory

## Related Work

- [Pull Request to mlr3extralearners](https://github.com/mlr-org/mlr3extralearners/pull/YOUR-PR-NUMBER)
- [GPyTorch Documentation](https://gpytorch.ai/)
- [mlr3 Book](https://mlr3book.mlr-org.com/)

## Author

Khalifa Seck - [GitHub](https://github.com/KhalifaSeck)

## License

MIT License
