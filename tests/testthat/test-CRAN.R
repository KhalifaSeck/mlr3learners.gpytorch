
library(reticulate)

tryCatch({
  use_condaenv("gpytorch", required = FALSE)
}, error = function(e) {
  message("GPyTorch environment not found")
})

test_that("regr.gpytorch basic functionality works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("gpytorch"), "GPyTorch not available")
  
  learner <- lrn("regr.gpytorch", n_iter = 30)
  task <- tsk("mtcars")
  
  expect_silent(learner$train(task))
  
  pred <- learner$predict(task)
  
  expect_true(!is.null(pred$response))
  expect_equal(length(pred$response), task$nrow)
  expect_true(all(!is.na(pred$response)))
  expect_true(is.numeric(pred$response))
})

test_that("regr.gpytorch learns non-trivial patterns", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("gpytorch"), "GPyTorch not available")
  
  learner_gp <- lrn("regr.gpytorch", n_iter = 30)
  learner_baseline <- lrn("regr.featureless")
  task <- tsk("mtcars")
  
  learner_gp$train(task)
  learner_baseline$train(task)
  
  pred_gp <- learner_gp$predict(task)
  pred_baseline <- learner_baseline$predict(task)
  
  mse_gp <- pred_gp$score(msr("regr.mse"))
  mse_baseline <- pred_baseline$score(msr("regr.mse"))
  
  expect_true(mse_gp < mse_baseline)
})

test_that("regr.gpytorch hyperparameters work", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("gpytorch"), "GPyTorch not available")
  
  learner <- lrn("regr.gpytorch", kernel = "matern", n_iter = 20, lr = 0.05)
  task <- tsk("mtcars")
  
  expect_silent(learner$train(task))
  
  pred <- learner$predict(task)
  
  expect_true(!is.null(pred$response))
  expect_equal(length(pred$response), task$nrow)
})

test_that("regr.gpytorch standard error prediction works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("gpytorch"), "GPyTorch not available")
  
  learner <- lrn("regr.gpytorch", predict_type = "se", n_iter = 30)
  task <- tsk("mtcars")
  
  learner$train(task)
  pred <- learner$predict(task)
  
  expect_true(!is.null(pred$response))
  expect_true(!is.null(pred$se))
  expect_equal(length(pred$se), task$nrow)
  expect_true(all(pred$se > 0))
})

