library(reticulate)
use_condaenv("gpytorch", required = TRUE)

library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(ggplot2)
library(data.table)
library(MASS)

devtools::load_all()

set.seed(42)
n_folds <- 5
max_k <- 30

data(airquality)
df_airquality <- as.data.table(na.omit(airquality))

data(Boston, package = "MASS")
df_boston <- as.data.table(Boston)

task_list <- list(
  airquality = TaskRegr$new("airquality", df_airquality, target = "Ozone"),
  boston = TaskRegr$new("boston", df_boston, target = "medv")
)

learner_gpytorch <- lrn("regr.gpytorch", n_iter = 50, id = "gpytorch")
learner_featureless <- lrn("regr.featureless", id = "featureless")
learner_glmnet <- lrn("regr.cv_glmnet", id = "glmnet")

knn_learner <- lrn("regr.kknn")
knn_learner$param_set$values$k <- to_tune(1, max_k)

kfoldcv <- rsmp("cv")
kfoldcv$param_set$values$folds <- 3

learner_knn <- auto_tuner(
  learner = knn_learner,
  tuner = tnr("grid_search"),
  resampling = kfoldcv,
  measure = msr("regr.mse")
)
learner_knn$id <- "knn"

learner.list <- list(
  gpytorch = learner_gpytorch,
  featureless = learner_featureless,
  glmnet = learner_glmnet,
  knn = learner_knn
)

rsmp_cv <- rsmp("cv", folds = n_folds)

bench.grid <- benchmark_grid(
  task_list,
  learner.list,
  rsmp_cv
)

bench.result <- tryCatch(
  suppressWarnings(benchmark(bench.grid, store_models = TRUE)),
  error = function(e) {
    benchmark(bench.grid, store_models = FALSE)
  }
)

if (is.null(bench.result)) {
  stop("Benchmark failed due to numerical instability.")
}

test_measure_list <- msrs("regr.mse")
score_dt <- bench.result$score(test_measure_list)

gg <- ggplot(score_dt, aes(x = regr.mse, y = learner_id)) +
  geom_point(size = 3, alpha = 0.7) +
  facet_grid(task_id ~ ., scales = "free_x") +
  labs(
    x = "Mean Squared Error (MSE)", 
    y = "Algorithm",
    title = "Prediction Error in 5-Fold Cross-Validation"
  ) +
  theme_bw()

print(gg)
ggsave("benchmark_results.png", gg, width = 10, height = 6, dpi = 300)

mean_err <- score_dt[
  ,
  .(mean_mse = mean(regr.mse)),
  by = .(task_id, learner_id)
]

mean_err <- mean_err[order(task_id, mean_mse)]

print(mean_err)

for (dataset in unique(score_dt$task_id)) {
  gpytorch_mse <- mean(score_dt[task_id == dataset & learner_id == "gpytorch", regr.mse])
  featureless_mse <- mean(score_dt[task_id == dataset & learner_id == "featureless", regr.mse])
  
  improvement <- ((featureless_mse - gpytorch_mse) / featureless_mse) * 100
  
  cat(sprintf("Dataset: %s\n", dataset))
  cat(sprintf("  GPyTorch MSE: %.4f\n", gpytorch_mse))
  cat(sprintf("  Featureless MSE: %.4f\n", featureless_mse))
  cat(sprintf("  Improvement: %.2f%%\n", improvement))
  
  if (improvement > 0) {
    cat("  YES, GPyTorch learns non-trivial patterns\n\n")
  } else {
    cat("  No significant learning\n\n")
  }
}

for (dataset in unique(score_dt$task_id)) {
  cat(sprintf("Dataset: %s\n", dataset))
  
  perf <- mean_err[task_id == dataset][order(mean_mse)]
  
  for (i in 1:nrow(perf)) {
    cat(sprintf("  %d. %-15s MSE = %.4f\n", 
                i, 
                perf$learner_id[i], 
                perf$mean_mse[i]))
  }
  
  gpytorch_rank <- which(perf$learner_id == "gpytorch")
  cat(sprintf("  GPyTorch ranks #%d out of %d\n\n", gpytorch_rank, nrow(perf)))
}

validate_hyperparameters <- function(task, seed = 123) {
  set.seed(seed)
  train_idx <- sample(1:task$nrow, 0.7 * task$nrow)
  test_idx <- setdiff(1:task$nrow, train_idx)
  
  configs <- list(
    list(name = "Default (RBF)", params = list()),
    list(name = "Matern", params = list(kernel = "matern")),
    list(name = "RBF lr=0.05", params = list(lr = 0.05))
  )
  
  results <- data.table()
  
  for (config in configs) {
    learner <- lrn("regr.gpytorch", n_iter = 30)
    if (length(config$params) > 0) {
      learner$param_set$values <- config$params
    }
    
    train_success <- tryCatch({
      suppressWarnings(learner$train(task, row_ids = train_idx))
      TRUE
    }, error = function(e) {
      FALSE
    })
    
    if (!train_success) {
      next
    }
    
    pred_train <- learner$predict(task, row_ids = train_idx)
    pred_test <- learner$predict(task, row_ids = test_idx)
    
    mse_train <- pred_train$score(msr("regr.mse"))
    mse_test <- pred_test$score(msr("regr.mse"))
    
    results <- rbind(results, data.table(
      Configuration = config$name,
      MSE_Train = round(mse_train, 4),
      MSE_Test = round(mse_test, 4)
    ))
  }
  
  learner_baseline <- lrn("regr.featureless")
  learner_baseline$train(task, row_ids = train_idx)
  pred_train_baseline <- learner_baseline$predict(task, row_ids = train_idx)
  pred_test_baseline <- learner_baseline$predict(task, row_ids = test_idx)
  
  mse_train_baseline <- pred_train_baseline$score(msr("regr.mse"))
  mse_test_baseline <- pred_test_baseline$score(msr("regr.mse"))
  
  results <- rbind(results, data.table(
    Configuration = "Baseline",
    MSE_Train = round(mse_train_baseline, 4),
    MSE_Test = round(mse_test_baseline, 4)
  ))
  
  if (nrow(results) > 1) {
    results[, Improvement := ifelse(
      Configuration == "Baseline",
      "-",
      sprintf("%.1f%%", (mse_test_baseline - MSE_Test) / mse_test_baseline * 100)
    )]
  }
  
  return(results)
}

results_airquality <- validate_hyperparameters(task_list$airquality, seed = 123)
print(results_airquality)

results_boston <- validate_hyperparameters(task_list$boston, seed = 456)
print(results_boston)

score_dt_export <- score_dt[, .(task_id, learner_id, iteration, regr.mse)]
fwrite(score_dt_export, "benchmark_detailed_results.csv")
fwrite(mean_err, "benchmark_summary.csv")
fwrite(results_airquality, "validation_airquality.csv")
fwrite(results_boston, "validation_boston.csv")