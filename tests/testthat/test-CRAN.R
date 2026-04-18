test_that("regr.gpytorch works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("gpytorch"))
  
  learner <- lrn("regr.gpytorch", n_iter = 30)
  task <- tsk("mtcars")
  
  learner$train(task)
  pred <- learner$predict(task)
  
  expect_true(!is.null(pred$response))
  expect_equal(length(pred$response), task$nrow)
})

test_that("regr.gpytorch hyperparameters work", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("gpytorch"))
  
  learner <- lrn("regr.gpytorch", kernel = "matern", n_iter = 20, lr = 0.05)
  task <- tsk("mtcars")
  
  learner$train(task)
  pred <- learner$predict(task)
  
  expect_true(!is.null(pred$response))
})

test_that("regr.gpytorch se prediction works", {
  skip_if_not_installed("reticulate")
  skip_if(!reticulate::py_module_available("gpytorch"))
  
  learner <- lrn("regr.gpytorch", predict_type = "se", n_iter = 30)
  task <- tsk("mtcars")
  
  learner$train(task)
  pred <- learner$predict(task)
  
  expect_true(!is.null(pred$response))
  expect_true(!is.null(pred$se))
  expect_equal(length(pred$se), task$nrow)
})


