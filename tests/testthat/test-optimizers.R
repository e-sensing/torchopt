library(torchopt)
euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))
test_that("adamw optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- torchopt::test_optim_valid(
        optim = torchopt::optim_adamw,
        opt_hparams = list(lr = 0.05),
        steps = 400,
        test_fn = "beale"
    )
    xy_opt <- c(3, 0.5)
    xy_0 <- c(xy[[1]][1], xy[[2]][1])
    xy_200 <- c(xy[[1]][200], xy[[2]][200])

    diff0 <- euc.dist(xy_opt, xy_0)
    diff1 <- euc.dist(xy_opt, xy_200)

    expect_true(diff1 < diff0)
})

test_that("adabelief optimizer", {
    testthat::skip_on_cran()
    set.seed(42)
    xy <- test_optim_valid(
        optim = optim_adabelief,
        opt_hparams = list(lr = 0.5),
        steps = 400,
        test_fn = "beale"
    )
    xy_opt <- c(3, 0.5)
    xy_0 <- c(xy[[1]][1], xy[[2]][1])
    xy_200 <- c(xy[[1]][200], xy[[2]][200])

    diff0 <- euc.dist(xy_opt, xy_0)
    diff1 <- euc.dist(xy_opt, xy_200)

    expect_true(diff1 < diff0)

})

test_that("adabound optimizer", {
    testthat::skip_on_cran()
    set.seed(22)
    xy <- test_optim_valid(
        optim = optim_adabound,
        opt_hparams = list(lr = 0.5),
        steps = 400,
        test_fn = "beale"
    )
    xy_opt <- c(3, 0.5)
    xy_0 <- c(xy[[1]][1], xy[[2]][1])
    xy_200 <- c(xy[[1]][200], xy[[2]][200])

    diff0 <- euc.dist(xy_opt, xy_0)
    diff1 <- euc.dist(xy_opt, xy_200)

    expect_true(diff1 < diff0)

})
test_that("madgrad optimizer", {
    testthat::skip_on_cran()
    set.seed(256)
    xy <- test_optim_valid(
        optim = optim_madgrad,
        opt_hparams = list(lr = 0.1),
        steps = 400,
        test_fn = "beale"
    )
    xy_opt <- c(3, 0.5)
    xy_0 <- c(xy[[1]][1], xy[[2]][1])
    xy_200 <- c(xy[[1]][200], xy[[2]][200])

    diff0 <- euc.dist(xy_opt, xy_0)
    diff1 <- euc.dist(xy_opt, xy_200)

    expect_true(diff1 < diff0)

})
test_that("yogi optimizer", {
    testthat::skip_on_cran()
    set.seed(66)
    xy <- test_optim_valid(
        optim = optim_yogi,
        opt_hparams = list(lr = 0.1),
        steps = 400,
        test_fn = "beale"
    )
    xy_opt <- c(3, 0.5)
    xy_0 <- c(xy[[1]][1], xy[[2]][1])
    xy_200 <- c(xy[[1]][200], xy[[2]][200])

    diff0 <- euc.dist(xy_opt, xy_0)
    diff1 <- euc.dist(xy_opt, xy_200)

    expect_true(diff1 < diff0)

})
