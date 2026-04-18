

#' @title GPyTorch Gaussian Process Regression Learner
#' @author Khalifa Seck
#' @name mlr_learners_regr.gpytorch
#'
#' @description
#' Gaussian Process regression using GPyTorch (PyTorch-based).
#' Requires Python with torch and gpytorch installed.
#'
#' @references
#' Gardner J, Pleiss G, Weinberger KQ, Bindel D, Wilson AG (2018).
#' "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration."
#' In Advances in Neural Information Processing Systems.
#'
#' @export
LearnerRegrGPyTorch <- R6::R6Class("LearnerRegrGPyTorch",
  inherit = mlr3::LearnerRegr,
  
  public = list(
    #' @description
    #' Creates a new instance of this R6 class.
    initialize = function() {
      ps <- paradox::ps(
        kernel = paradox::p_fct(levels = c("rbf", "matern"), default = "rbf", tags = "train"),
        lr = paradox::p_dbl(lower = 0.001, upper = 1, default = 0.1, tags = "train"),
        n_iter = paradox::p_int(lower = 10, upper = 500, default = 50, tags = "train"),
        device = paradox::p_fct(levels = c("auto", "cpu", "cuda"), default = "auto", tags = "train")
      )
    
      
      super$initialize(
        id = "regr.gpytorch",
        packages = c("reticulate", "mlr3learners.gpytorch"),
        feature_types = c("integer", "numeric"),
        predict_types = c("response", "se"),
        param_set = ps,
        properties = character(0),
        man = "mlr3learners.gpytorch::mlr_learners_regr.gpytorch",
        label = "Gaussian Process (GPyTorch)"
      )
    }
  ),
  
  private = list(
    .train = function(task) {
      pv <- self$param_set$get_values(tags = "train")
      
      X <- as.matrix(task$data(cols = task$feature_names))
      y <- as.numeric(task$truth())
      
      python_path <- system.file("python", "gp_model.py", package = "mlr3learners.gpytorch")
      reticulate::source_python(python_path)
      
      kernel <- pv$kernel
      if (is.null(kernel)) kernel <- "rbf"
      
      lr <- pv$lr
      if (is.null(lr)) lr <- 0.1
      
      n_iter <- pv$n_iter
      if (is.null(n_iter)) n_iter <- 50
      
      device <- pv$device
      if (is.null(device)) device <- "auto"
      
      model <- GPyTorchWrapper(kernel = kernel, lr = lr, n_iter = n_iter, device = device)
      model$fit(X, y)
      
      return(model)
    },
    .predict = function(task) {
      X_new <- as.matrix(task$data(cols = task$feature_names))
      
      pred <- self$model$predict(X_new)
      
      if (self$predict_type == "se") {
        list(
          response = pred$mean,
          se = sqrt(pred$variance)
        )
      } else {
        list(response = pred$mean)
      }
    }
  )
)

mlr3::mlr_learners$add("regr.gpytorch", LearnerRegrGPyTorch)


