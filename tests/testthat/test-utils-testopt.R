test_that("can use custom functions with test_opt", {
    testthat::skip_on_cran()
    set.seed(1)
    expect_error(regexp = NA,{
        test_optim(
            optim = optim_adamw,
            test_fn = list(beale, domain_beale),
            opt_hparams = list(lr = 0.05),
            steps = 100,
            plot_each_step = TRUE
        )
    })
})
