#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @importFrom graphics contour
#' @importFrom graphics image
#' @importFrom graphics lines
#' @importFrom graphics points
#' @importFrom grDevices hcl.colors
#' @importFrom stats runif
## usethis namespace: end
NULL

.onLoad <- function(libname, pkgname) {
    if (!torch::torch_is_installed())
        torch::install_torch()
}
# Include the following global variables
utils::globalVariables(c("self", "super", "ctx", "private"))

