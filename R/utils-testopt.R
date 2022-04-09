
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

#' @title Test optimization function
#'
#' @name test_optim
#'
#' @author Rolf Simoes, \email{rolf.simoes@@inpe.br}
#'
#' @description
#' `test_optim()` function is useful to visualize how optimizers solve the
#' minimization problem by showing the convergence path using a test function.
#' User can choose any test optimization
#' [functions](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
#' provided by `torchopt`:
#'
#' - `"beale"`, `"booth"`, `"bukin_n6"`, `"easom"`, `"goldstein_price"`,
#' `"himmelblau"`, `"levi_n13"`, `"matyas"`, `"rastrigin"`,
#' `"rosenbrock"`, and `"sphere"`.
#'
#' Besides these functions, users can pass any function that receives two
#' numerical values and returns a scalar.
#'
#' Optimization functions are useful to evaluate characteristics of optimization
#' algorithms, such as convergence rate, precision, robustness, and performance.
#' These functions give an idea about the different situations that optimization
#' algorithms can face.
#'
#' Function `test_function()` plot the 2D-space of a test optimization function.
#'
#' @param optim          Torch optimizer function.
#' @param ...            Additional parameters (passed to `image` function).
#' @param x0,y0          Initial X and Y coordinates
#'   (default random between `-5` and `5`).
#' @param opt_hparams    A list with optimizer initialize parameters
#'   (default `list(lr = 0.01)`).
#' @param test_fn        A test function (default `"beale"`).
#' @param steps          Number of steps to run (default `100`).
#' @param pt_start_color Starting point color (default `"#5050FF7F"`)
#' @param pt_end_color   Ending point color (default `"#FF5050FF"`)
#' @param ln_color       Line path color (default `"#FF0000FF"`)
#' @param ln_weight      Line path weight (default `2`)
#' @param bg_x_lim       Background X limits (default `NULL`)
#' @param bg_y_lim       Background Y limits (default `NULL`)
#' @param bg_xy_breaks   Background X and Y resolution (default `100`)
#' @param bg_z_breaks    Background Z resolution (default `32`)
#' @param bg_palette     Background palette (default `"viridis"`)
#' @param ct_levels      Contour levels (default `10`)
#' @param ct_labels      Should show contour labels? (default `FALSE`)
#' @param ct_color       Contour color (default `"#FFFFFF7F"`)
#' @param plot_each_step Should output each step? (default `FALSE`)
#'
#' @examples
#' # plot test function 2D-space
#' test_function("rosenbrock")
#'
#' # plot optimization path of a initially random stating point
#' test_optim(optim_madgrad)
#'
NULL

#' @rdname test_optim
#' @export
test_function <- function(test_fn, ...,
                          bg_x_lim = c(-5, 5),
                          bg_y_lim = c(-5, 5),
                          bg_xy_breaks = 100,
                          bg_z_breaks = 32,
                          bg_palette = "viridis",
                          ct_levels = 10,
                          ct_labels = TRUE,
                          ct_color = "#FFFFFF7F") {

    # pre-conditions
    if (is.character(test_fn)) {
        if (!exists(test_fn,
                    envir = asNamespace("torchopt"),
                    inherits = FALSE)) {
            stop("invalid 'test_fn' param.", call. = FALSE)
        }
        # get gradient function
        test_fn <- get(test_fn,
                       envir = asNamespace("torchopt"),
                       inherits = FALSE)
    }
    if (!is.function(test_fn)) {
        stop("invalid 'test_fn' param.", call. = FALSE)
    }

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
    if (ct_levels > 0) {
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

}

#' @rdname test_optim
#' @export
test_optim <- function(optim, ...,
                       x0 = runif(1, -5, 5),
                       y0 = runif(1, -5, 5),
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
                       ct_color = "#FFFFFF7F",
                       plot_each_step = FALSE) {

    # pre-conditions
    if (!inherits(optim, "function")) {
        stop("invalid 'opt' param.", call. = FALSE)
    }
    if (is.character(test_fn)) {
        if (!exists(test_fn,
                    envir = asNamespace("torchopt"),
                    inherits = FALSE)) {
            stop("invalid 'test_fn' param.", call. = FALSE)
        }
        # get gradient function
        test_fn <- get(test_fn,
                       envir = asNamespace("torchopt"),
                       inherits = FALSE)
    }
    if (!is.function(test_fn)) {
        stop("invalid 'test_fn' param.", call. = FALSE)
    }

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
        if (ct_levels > 0) {
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
