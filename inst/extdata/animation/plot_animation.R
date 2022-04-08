library(madgrad)
library(torch)
library(gifski)

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

torch::torch_manual_seed(42)

gifski::save_gif(
    test_optim(
        opt = torchopt::optim_madgrad,
        opt_hparams = list(lr = 0.1),
        test_fn = "beale",
        steps = 100,
        plot_each_step = TRUE
    ),
    delay = 0.1,
    loop = TRUE
)

#
# plot_bg <- function(grad_fn, ...,
#                     x_lim = c(-5, 5),
#                     y_lim = c(-5, 5),
#                     xy_breaks = 255,
#                     z_breaks = 255,
#                     palette = "viridis",
#                     contour_levels = 10,
#                     contour_labels = FALSE,
#                     contour_color = "#FFFFFF50") {
#
#     x <- seq(x_lim[[1]], x_lim[[2]], length.out = xy_breaks)
#     y <- seq(y_lim[[1]], y_lim[[2]], length.out = xy_breaks)
#     z <- outer(X = x, Y = y, FUN = grad_fn)
#
#     # plot background
#     image(
#         x = x,
#         y = y,
#         z = z,
#         col = hcl.colors(
#             n = z_breaks,
#             palette = palette
#         ),
#         ...
#     )
#     # plot contour
#     contour(
#         x = x,
#         y = y,
#         z = z,
#         nlevels = contour_levels,
#         drawlabels = contour_labels,
#         col = contour_color,
#         add = TRUE
#     )
# }
#
# plot_test_optim <- function(grad_data, ...,
#                             pt_start_color = "#5050FF7F",
#                             pt_end_color = "#FF5050FF",
#                             ln_color = "#FF0000FF",
#                             ln_weight = 2,
#                             bg_x_lim = NULL,
#                             bg_y_lim = NULL,
#                             bg_xy_breaks = 100,
#                             bg_z_breaks = 32,
#                             bg_palette = "viridis") {
#
#     stopifnot(inherits(grad_data, "test_optim"))
#     stopifnot(c("x", "y", "grad_fn") %in% names(grad_data))
#
#     steps <- length(grad_data$x)
#     grad_fn <- grad_data$grad_fn
#
#     if (is.null(bg_x_lim)) {
#         x_min <- min(grad_data$x)
#         x_max <- max(grad_data$x)
#         x_fac <- (x_max - x_min) * 0.1
#         if (steps == 1) x_fac <- 1
#         bg_x_lim <- c(x_min - x_fac, x_max + x_fac)
#     }
#
#     if (is.null(bg_y_lim)) {
#         y_min <- min(grad_data$y)
#         y_max <- max(grad_data$y)
#         y_fac <- (y_max - y_min) * 0.1
#         if (steps == 1) y_fac <- 1
#         bg_y_lim <- c(y_min - y_fac, y_max + y_fac)
#     }
#
#     # plot grad function background
#     plot_bg(
#         grad_fn = grad_fn,
#         x_lim = bg_x_lim,
#         y_lim = bg_y_lim,
#         xy_breaks = bg_xy_breaks,
#         z_breaks = bg_z_breaks,
#         palette = bg_palette,
#         ...
#     )
#
#     # staring point
#     points(
#         grad_data$x[1],
#         grad_data$y[1],
#         pch = 21,
#         bg = pt_start_color
#     )
#
#     # path line
#     lines(
#         grad_data$x,
#         grad_data$y,
#         lwd = ln_weight,
#         col = ln_color
#     )
#
#     # end point
#     points(
#         grad_data$x[steps],
#         grad_data$y[steps],
#         pch = 21,
#         bg = pt_end_color
#     )
# }

plot_grad_test(
    data,
    bg_x_lim = c(-5, 5),
    bg_y_lim = c(-5, 5),
    bg_z_breaks = 128,
    bg_xy_breaks = 255
)



gifski::save_gif(
    expr = {
        bg <- function(fn) {

            x1 <- (-100:100) * 0.1
            y1 <- (-100:100) * 0.1
            z <- outer(x1, y1, FUN = fn)
            image(
                x = x1,
                y = y1,
                z = z,
                col = hcl.colors(16),
                xlim = c(-10, 10),
                ylim = c(-10, 10)
            )
        }

        torch::torch_manual_seed(1)
        x <- torch::torch_tensor(-5, requires_grad = TRUE)
        y <- torch::torch_tensor(-2, requires_grad = TRUE)
        opt <- madgrad::optim_madgrad(params = list(x, y), lr = 0.1)

        for (i in 1:100) {

            opt$zero_grad()
            z <- f(x, y)
            z$backward()
            opt$step()
            bg(fn = f)
            points(x, y)
        }
    },
    delay = 0.1,
    loop = TRUE
)

