library(testthat)
library(torchopt)

if (Sys.getenv("TORCH_TEST", unset = 0) == 1) {
    test_check("torchopt")
}

