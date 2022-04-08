
beale <- function(x, y) {
    log((1.5 - x + x * y)^2 + (2.25 - x - x * y^2)^2 + (2.625 - x + x * y^3)^2)
}

booth <- function(x, y) {
    log((x + 2 * y - 7)^2 + (2 * x + y - 5)^2)
}

bukin_n6 <- function(x, y) {
    100 * sqrt(abs(y - 0.01 * x^2)) + 0.01 * abs(x + 10)
}

easom <- function(x, y) {
    -cos(x) * cos(y) * exp(-(x - pi)^2 - (y - pi)^2)
}

goldstein_price <- function(x, y) {
    log((1 + (x + y + 1)^2 *
             (19 - 14 * x + 3 * x^2 - 14 * y + 6 * x * y + 3 * y^2)) *
            (30 + (2 * x - 3 * y)^2 * (18 - 32 * x + 12 * x^2 + 48 *
                                           y - 36 * x * y + 27 * y^2)))
}

himmelblau <- function(x, y) {
    log((x^2 + y - 11)^2 + (x + y^2 - 7)^2)
}

levi_n13 <- function(x, y) {
    sin(3 * pi * x)^2 + (x - 1)^2 * (1 + sin(3 * pi * y)^2) +
        (y - 1)^2 * (1 + sin(2 * pi * y)^2)
}

matyas <- function(x, y) {
    log(0.26 * (x^2 + y^2) - 0.48 * x * y)
}

rastrigin <- function(x, y) {
    20 + (x^2 - 10 * cos(2 * pi * x)) + (y^2 - 10 * cos(2 * pi * y))
}

rosenbrock <- function(x, y) {
    log(100 * (y - x^2)^2 + (1 - x)^2)
}

sphere <- function(x, y) {
    x^2 + y^2
}

#' @export
test_function <- function(test_fn, ...,
                          bg_x_lim = c(-5, 5),
                          bg_y_lim = c(-5, 5),
                          bg_xy_breaks = 100,
                          bg_z_breaks = 32,
                          bg_palette = "viridis",
                          ct_levels = 10,
                          ct_labels = TRUE,
                          ct_color = "#FFFFFF50") {

    # get gradient function
    test_fn <-
        tryCatch({
            get(test_fn,
                envir = asNamespace("torchopt"),
                inherits = FALSE
            )
        },
        error = function(e) {
            stop("invalid 'test_fn' parameter.", call. = FALSE)
        })

    # prepare data for gradient plot
    x <- seq(bg_x_lim[[1]], bg_x_lim[[2]], length.out = bg_xy_breaks)
    y <- seq(bg_y_lim[[1]], bg_y_lim[[2]], length.out = bg_xy_breaks)
    z <- outer(X = x, Y = y, FUN = test_fn)

    # plot background
    image(
        x = x,
        y = y,
        z = z,
        col = hcl.colors(
            n = bg_z_breaks,
            palette = bg_palette
        ),
        ...
    )

    # plot contour
    contour(
        x = x,
        y = y,
        z = z,
        nlevels = ct_levels,
        drawlabels = ct_labels,
        col = ct_color,
        add = TRUE
    )

}

#' @export
test_optim <- function(opt, ...,
                       params = list(
                           torch::torch_randint(-5, 5, 1, requires_grad = TRUE),
                           torch::torch_randint(-5, 5, 1, requires_grad = TRUE)
                       ),
                       opt_hparams = list(lr = 0.01),
                       test_fn = "beale",
                       steps = 100,
                       pt_start_color = "#5050FF7F",
                       pt_end_color = "#FF5050FF",
                       ln_color = "#FF0000FF",
                       ln_weight = 2,
                       bg_x_lim = NULL,
                       bg_y_lim = NULL,
                       bg_xy_breaks = 100,
                       bg_z_breaks = 32,
                       bg_palette = "viridis",
                       ct_levels = 10,
                       ct_labels = FALSE,
                       ct_color = "#FFFFFF50",
                       plot_each_step = FALSE) {

    stopifnot(inherits(opt, "function"))

    opt <- do.call(opt, c(list(params = params), opt_hparams))

    stopifnot(length(opt$param_groups[[1]]$params) == 2)

    # get gradient function
    test_fn <- tryCatch({
        get(test_fn,
            envir = asNamespace("torchopt"),
            inherits = FALSE
        )
    },
    error = function(e) {
        stop("invalid 'test_fn' parameter.", call. = FALSE)
    })

    # get params
    x <- opt$param_groups[[1]]$params[[1]]
    y <- opt$param_groups[[1]]$params[[2]]

    # run optimizer
    x_steps <- numeric(steps)
    y_steps <- numeric(steps)
    for (i in seq_len(steps)) {
        x_steps[i] <- as.numeric(x)
        y_steps[i] <- as.numeric(y)
        opt$zero_grad()
        z <- test_fn(x, y)
        z$backward()
        opt$step()
    }

    # prepare plot
    # get x limits
    if (is.null(bg_x_lim)) {
        x_min <- min(x_steps)
        x_max <- max(x_steps)
        x_fac <- (x_max - x_min) * 0.1
        if (steps == 1) x_fac <- 1
        bg_x_lim <- c(x_min - x_fac, x_max + x_fac)
    }

    # get y limits
    if (is.null(bg_y_lim)) {
        y_min <- min(y_steps)
        y_max <- max(y_steps)
        y_fac <- (y_max - y_min) * 0.1
        if (steps == 1) y_fac <- 1
        bg_y_lim <- c(y_min - y_fac, y_max + y_fac)
    }

    # prepare data for gradient plot
    x <- seq(bg_x_lim[[1]], bg_x_lim[[2]], length.out = bg_xy_breaks)
    y <- seq(bg_y_lim[[1]], bg_y_lim[[2]], length.out = bg_xy_breaks)
    z <- outer(X = x, Y = y, FUN = test_fn)

    plot_from_step <- steps
    if (plot_each_step) {
        plot_from_step <- 1
    }

    for (step in seq(plot_from_step, steps, 1)) {

        # plot background
        image(
            x = x,
            y = y,
            z = z,
            col = hcl.colors(
                n = bg_z_breaks,
                palette = bg_palette
            ),
            ...
        )

        # plot contour
        contour(
            x = x,
            y = y,
            z = z,
            nlevels = ct_levels,
            drawlabels = ct_labels,
            col = ct_color,
            add = TRUE
        )

        # plot starting point
        points(
            x_steps[1],
            y_steps[1],
            pch = 21,
            bg = pt_start_color
        )

        # plot path line
        lines(
            x_steps[seq_len(step)],
            y_steps[seq_len(step)],
            lwd = ln_weight,
            col = ln_color
        )

        # plot end point
        points(
            x_steps[step],
            y_steps[step],
            pch = 21,
            bg = pt_end_color
        )
    }
}
