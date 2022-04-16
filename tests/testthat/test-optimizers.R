library(torchopt)
beale <- function(x, y) {
    log((1.5 - x + x * y)^2 + (2.25 - x - x * y^2)^2 + (2.625 - x + x * y^3)^2)
}
test_that("adamw optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- torchopt::test_optim_valid(
        optim = torchopt::optim_adamw,
        opt_hparams = list(lr = 0.05),
        steps = 400,
        test_fn = "beale"
    )

    x0 <- xy[[1]][1]
    y0 <- xy[[2]][1]
    x400 <- xy[[1]][400]
    y400 <- xy[[2]][400]
    test_fn0 <- beale(x0, y0)
    test_fn400 <- beale(x400, y400)

    expect_true(test_fn0 > test_fn400)
})

test_that("adabelief optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- test_optim_valid(
        optim = optim_adabelief,
        opt_hparams = list(lr = 0.5),
        steps = 400,
        test_fn = "beale"
    )
    test_fn0 <- beale(xy[[1]][1], xy[[2]][1])
    test_fn400 <- beale(xy[[1]][400], xy[[2]][400])

    expect_true(test_fn0 > test_fn400)
})

test_that("adabound optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- test_optim_valid(
        optim = optim_adabound,
        opt_hparams = list(lr = 0.5),
        steps = 400,
        test_fn = "beale"
    )
    test_fn0 <- beale(xy[[1]][1], xy[[2]][1])
    test_fn400 <- beale(xy[[1]][400], xy[[2]][400])

    expect_true(test_fn0 > test_fn400)

})
test_that("madgrad optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- test_optim_valid(
        optim = optim_madgrad,
        opt_hparams = list(lr = 0.1),
        steps = 400,
        test_fn = "beale"
    )
    test_fn0 <- beale(xy[[1]][1], xy[[2]][1])
    test_fn400 <- beale(xy[[1]][400], xy[[2]][400])

    expect_true(test_fn0 > test_fn400)

})

test_that("nadam optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- test_optim_valid(
        optim = optim_nadam,
        opt_hparams = list(lr = 0.1),
        steps = 400,
        test_fn = "beale"
    )
    test_fn0 <- beale(xy[[1]][1], xy[[2]][1])
    test_fn400 <- beale(xy[[1]][400], xy[[2]][400])

    expect_true(test_fn0 > test_fn400)

})
test_that("qhadam optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- test_optim_valid(
        optim = optim_qhadam,
        opt_hparams = list(lr = 0.1),
        steps = 400,
        test_fn = "beale"
    )
    test_fn0 <- beale(xy[[1]][1], xy[[2]][1])
    test_fn400 <- beale(xy[[1]][400], xy[[2]][400])

    expect_true(test_fn0 > test_fn400)

})
test_that("radam optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- test_optim_valid(
        optim = optim_radam,
        opt_hparams = list(lr = 0.1),
        steps = 400,
        test_fn = "beale"
    )
    test_fn0 <- beale(xy[[1]][1], xy[[2]][1])
    test_fn400 <- beale(xy[[1]][400], xy[[2]][400])

    expect_true(test_fn0 > test_fn400)

})
test_that("swats optimizer", {
    testthat::skip_on_cran()
    set.seed(234)
    xy <- test_optim_valid(
        optim = optim_swats,
        opt_hparams = list(lr = 0.1),
        steps = 400,
        test_fn = "beale"
    )
    test_fn0 <- beale(xy[[1]][1], xy[[2]][1])
    test_fn400 <- beale(xy[[1]][400], xy[[2]][400])

    expect_true(test_fn0 > test_fn400)

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
    test_fn0 <- beale(xy[[1]][1], xy[[2]][1])
    test_fn400 <- beale(xy[[1]][400], xy[[2]][400])

    expect_true(test_fn0 > test_fn400)

})
