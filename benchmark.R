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

py_run_string("import torch")
cuda_available <- py_eval("torch.cuda.is_available()")

cat("CUDA available:", cuda_available, "\n\n")

if (cuda_available) {
  devices_to_test <- c("cpu", "cuda")
} else {
  devices_to_test <- c("cpu")
  cat("GPU not available, using CPU only\n\n")
}

all_learners <- list()
device_results <- list()

for (dev in devices_to_test) {
  learner_gpytorch <- lrn("regr.gpytorch", n_iter = 50, device = dev, id = paste0("gpytorch_", dev))
  all_learners[[paste0("gpytorch_", dev)]] <- learner_gpytorch
}

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

all_learners[["featureless"]] <- learner_featureless
all_learners[["glmnet"]] <- learner_glmnet
all_learners[["knn"]] <- learner_knn

rsmp_cv <- rsmp("cv", folds = n_folds)

bench.grid <- benchmark_grid(
  task_list,
  all_learners,
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

score_dt[, device := ifelse(grepl("_cpu$", learner_id), "CPU",
                      ifelse(grepl("_cuda$", learner_id), "GPU", "N/A"))]

score_dt[, learner_base := gsub("_cpu$|_cuda$", "", learner_id)]

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
  by = .(task_id, learner_id, device)
]

mean_err <- mean_err[order(task_id, mean_mse)]

print(mean_err)

for (dataset in unique(score_dt$task_id)) {
  cat(sprintf("Dataset: %s\n", dataset))
  
  perf <- mean_err[task_id == dataset][order(mean_mse)]
  
  for (i in 1:nrow(perf)) {
    cat(sprintf("  %d. %-20s (%-3s) MSE = %.4f\n", 
                i, 
                perf$learner_id[i],
                perf$device[i],
                perf$mean_mse[i]))
  }
  
}

for (dataset in unique(score_dt$task_id)) {
  gpytorch_results <- mean_err[task_id == dataset & grepl("gpytorch", learner_id)]
  featureless_mse <- mean_err[task_id == dataset & learner_id == "featureless", mean_mse]
  
  cat(sprintf("Dataset: %s\n", dataset))
  
  for (i in 1:nrow(gpytorch_results)) {
    gp_mse <- gpytorch_results$mean_mse[i]
    improvement <- ((featureless_mse - gp_mse) / featureless_mse) * 100
    
    cat(sprintf("  %s (%s): %.4f MSE, %.2f%% improvement over baseline\n",
                gpytorch_results$learner_id[i],
                gpytorch_results$device[i],
                gp_mse,
                improvement))
  }
  
}

if (cuda_available) {
  
  for (dataset in unique(score_dt$task_id)) {
    cpu_mse <- mean_err[task_id == dataset & learner_id == "gpytorch_cpu", mean_mse]
    gpu_mse <- mean_err[task_id == dataset & learner_id == "gpytorch_cuda", mean_mse]
    
    cat(sprintf("Dataset: %s\n", dataset))
    cat(sprintf("  CPU MSE: %.4f\n", cpu_mse))
    cat(sprintf("  GPU MSE: %.4f\n", gpu_mse))
    cat(sprintf("  Difference: %.4f\n\n", abs(cpu_mse - gpu_mse)))
  }
}

validate_hyperparameters <- function(task, seed = 123, device = "auto") {
  set.seed(seed)
  train_idx <- sample(1:task$nrow, 0.7 * task$nrow)
  test_idx <- setdiff(1:task$nrow, train_idx)
  
  configs <- list(
    list(name = "Default (RBF)", params = list(device = device)),
    list(name = "Matern", params = list(kernel = "matern", device = device)),
    list(name = "RBF lr=0.05", params = list(lr = 0.05, device = device))
  )
  
  results <- data.table()
  
  for (config in configs) {
    learner <- lrn("regr.gpytorch", n_iter = 30)
    if (length(config$params) > 0) {
      learner$param_set$values <- config$params
    }
    
    train_success <- tryCatch({
      start_time <- Sys.time()
      suppressWarnings(learner$train(task, row_ids = train_idx))
      end_time <- Sys.time()
      train_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      list(success = TRUE, time = train_time)
    }, error = function(e) {
      list(success = FALSE, time = NA)
    })
    
    if (!train_success$success) {
      next
    }
    
    pred_train <- learner$predict(task, row_ids = train_idx)
    pred_test <- learner$predict(task, row_ids = test_idx)
    
    mse_train <- pred_train$score(msr("regr.mse"))
    mse_test <- pred_test$score(msr("regr.mse"))
    
    results <- rbind(results, data.table(
      Configuration = config$name,
      Device = device,
      MSE_Train = round(mse_train, 4),
      MSE_Test = round(mse_test, 4),
      Train_Time_sec = round(train_success$time, 4)
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
    Device = "N/A",
    MSE_Train = round(mse_train_baseline, 4),
    MSE_Test = round(mse_test_baseline, 4),
    Train_Time_sec = NA
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

validation_results <- list()

for (dev in devices_to_test) {
  cat(sprintf("\nDevice: %s\n", toupper(dev)))
  results_airquality <- validate_hyperparameters(task_list$airquality, seed = 123, device = dev)
  validation_results[[paste0("airquality_", dev)]] <- results_airquality
  print(results_airquality)
}


for (dev in devices_to_test) {
  cat(sprintf("\nDevice: %s\n", toupper(dev)))
  results_boston <- validate_hyperparameters(task_list$boston, seed = 456, device = dev)
  validation_results[[paste0("boston_", dev)]] <- results_boston
  print(results_boston)
}

score_dt_export <- score_dt[, .(task_id, learner_id, device, iteration, regr.mse)]
fwrite(score_dt_export, "benchmark_detailed_results.csv")
fwrite(mean_err, "benchmark_summary.csv")

for (name in names(validation_results)) {
  fwrite(validation_results[[name]], paste0("validation_", name, ".csv"))
}

for (name in names(validation_results)) {
  cat(sprintf("  validation_%s.csv\n", name))
}