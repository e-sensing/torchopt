library(torchopt)
beale <- function(x, y) {
    log((1.5 - x + x * y)^2 + (2.25 - x - x * y^2)^2 + (2.625 - x + x * y^3)^2)
}
test_optim_valid <- function(optim,
                             opt_hparams = list(lr = 0.01),
                             test_fn = "beale",
                             steps = 100) {

    # get starting points
    domain_fn <- get(paste0("domain_",test_fn),
                     envir = asNamespace("torchopt"),
                     inherits = FALSE)
    # get gradient function
    test_fn <- get(test_fn,
                   envir = asNamespace("torchopt"),
                   inherits = FALSE)

    # starting point
    dom <- domain_fn()
    x0 <- dom[["x0"]]
    y0 <- dom[["y0"]]

    # create tensor
    x <- torch::torch_tensor(x0, requires_grad = TRUE)
    y <- torch::torch_tensor(y0, requires_grad = TRUE)

    # instantiate optimizer
    optim <- do.call(optim, c(list(params = list(x, y)), opt_hparams))

    # run optimizer
    x_steps <- numeric(steps)
    y_steps <- numeric(steps)
    for (i in seq_len(steps)) {
        x_steps[i] <- as.numeric(x)
        y_steps[i] <- as.numeric(y)
        optim$zero_grad()
        z <- test_fn(x, y)
        z$backward()
        optim$step()
    }
    return(list(x_steps = x_steps,
                y_steps = y_steps))
}
test_that("adamw optimizer", {
    testthat::skip_on_cran()
    set.seed(12345)
    xy <- test_optim_valid(
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
