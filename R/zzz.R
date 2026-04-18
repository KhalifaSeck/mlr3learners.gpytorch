gpytorch_module <- NULL

.onLoad <- function(libname, pkgname) {
  if (reticulate::py_module_available("gpytorch")) {
    python_path <- system.file("python", "gp_model.py", package = pkgname)
    reticulate::source_python(python_path)
    gpytorch_module <<- reticulate::import("gpytorch")
  }
}