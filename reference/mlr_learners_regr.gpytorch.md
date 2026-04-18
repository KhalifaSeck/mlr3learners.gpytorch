# GPyTorch Gaussian Process Regression Learner

Gaussian Process regression using GPyTorch (PyTorch-based). Requires
Python with torch and gpytorch installed.

## References

Gardner J, Pleiss G, Weinberger KQ, Bindel D, Wilson AG (2018).
"GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU
Acceleration." In Advances in Neural Information Processing Systems.

## Author

Khalifa Seck

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`mlr3::LearnerRegr`](https://mlr3.mlr-org.com/reference/LearnerRegr.html)
-\> `LearnerRegrGPyTorch`

## Methods

### Public methods

- [`LearnerRegrGPyTorch$new()`](#method-LearnerRegrGPyTorch-new)

- [`LearnerRegrGPyTorch$clone()`](#method-LearnerRegrGPyTorch-clone)

Inherited methods

- [`mlr3::Learner$base_learner()`](https://mlr3.mlr-org.com/reference/Learner.html#method-base_learner)
- [`mlr3::Learner$configure()`](https://mlr3.mlr-org.com/reference/Learner.html#method-configure)
- [`mlr3::Learner$encapsulate()`](https://mlr3.mlr-org.com/reference/Learner.html#method-encapsulate)
- [`mlr3::Learner$format()`](https://mlr3.mlr-org.com/reference/Learner.html#method-format)
- [`mlr3::Learner$help()`](https://mlr3.mlr-org.com/reference/Learner.html#method-help)
- [`mlr3::Learner$predict()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict)
- [`mlr3::Learner$predict_newdata()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict_newdata)
- [`mlr3::Learner$print()`](https://mlr3.mlr-org.com/reference/Learner.html#method-print)
- [`mlr3::Learner$reset()`](https://mlr3.mlr-org.com/reference/Learner.html#method-reset)
- [`mlr3::Learner$selected_features()`](https://mlr3.mlr-org.com/reference/Learner.html#method-selected_features)
- [`mlr3::Learner$train()`](https://mlr3.mlr-org.com/reference/Learner.html#method-train)
- [`mlr3::LearnerRegr$predict_newdata_fast()`](https://mlr3.mlr-org.com/reference/LearnerRegr.html#method-predict_newdata_fast)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this R6 class.

#### Usage

    LearnerRegrGPyTorch$new()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerRegrGPyTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
