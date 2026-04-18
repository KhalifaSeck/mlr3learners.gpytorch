# mlr3learners.gpytorch

Gaussian Process regression learner for mlr3 using GPyTorch (PyTorch-based).

## Installation

```r
remotes::install_github("KhalifaSeck/mlr3learners.gpytorch")
```

## Requirements

- Python >= 3.7
- PyTorch
- GPyTorch

Install Python dependencies:

```bash
conda create -n gpytorch python=3.10
conda activate gpytorch
conda install pytorch -c pytorch
pip install gpytorch
```

## Usage

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

## Hyperparameters

- `kernel`: "rbf" or "matern" (default: "rbf")
- `lr`: Learning rate (default: 0.1, range: 0.001-1)
- `n_iter`: Number of training iterations (default: 50, range: 10-500)

## Benchmark Results

Performance was evaluated using Mean Squared Error (MSE) in 5-fold cross-validation.

| Dataset | GPyTorch | KNN | CV-Glmnet | Featureless |
|---------|----------|-----|-----------|-------------|
| airquality | **398.58** | 404.84 | 563.16 | 1107.45 |
| boston | **9.68** | 18.29 | 28.29 | 84.70 |

**Note**: Lower values indicate better performance. **Bold values** represent the best learner for each task.

### Analysis

**Does GPyTorch learn non-trivial patterns?**

- **airquality**: 64.01% improvement over baseline
- **boston**: 88.57% improvement over baseline

**YES**, GPyTorch learns non-trivial patterns on both datasets.

**Is GPyTorch competitive with other algorithms?**

- **airquality**: GPyTorch ranks **#1 out of 4** algorithms
- **boston**: GPyTorch ranks **#1 out of 4** algorithms

**GPyTorch achieves the best performance on both datasets.**

## Feature Scaling Validation

Automatic feature scaling to [0,1] is correctly applied during both training and prediction.

### Validation Results (70% train / 30% test split)

**airquality:**

| Configuration | MSE Train | MSE Test | Improvement |
|---------------|-----------|----------|-------------|
| Default (RBF) | 182.43 | 376.56 | 53.2% |
| Matern | 128.60 | 366.98 | 54.4% |
| RBF lr=0.05 | 163.68 | 377.97 | 53.1% |
| Baseline | 1229.26 | 805.08 | - |

**boston:**

| Configuration | MSE Train | MSE Test | Improvement |
|---------------|-----------|----------|-------------|
| Default (RBF) | 3.53 | 9.98 | 86.7% |
| Matern | 1.68 | 9.93 | 86.8% |
| RBF lr=0.05 | 3.56 | 9.92 | 86.8% |
| Baseline | 88.45 | 75.28 | - |

**Key observations:**
- MSE Train < MSE Test indicates proper scaling and no overfitting
- MSE Test << MSE Baseline shows excellent generalization
- Improvement > 50% on both datasets
- Hyperparameters allow fine-tuning of performance

## Technical Features

- **Automatic feature normalization**: Features are automatically normalized as required by GPyTorch
- **GPU/CPU support**: Runs on both GPU and CPU
- **Numerical stability**: Uses Gaussian likelihood with nugget for stability

## Related Work

- [Course wiki](https://github.com/tdhock/2026-01-aa-grande-echelle/wiki/projets)
- [GPyTorch](https://gpytorch.ai/)
- [mlr3](https://mlr3.mlr-org.com/)
- [PyTorch](https://pytorch.org/)

## Author

**Khalifa SECK** - [GitHub](https://github.com/KhalifaSeck)

## License

MIT License