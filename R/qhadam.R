#' @title QHAdam optimization algorithm
#'
#' @name optim_qhadam
#'
#' @author Gilberto Camara, \email{gilberto.camara@@inpe.br}
#' @author Daniel Falbel, \email{daniel.falble@@gmail.com}
#' @author Rolf Simoes, \email{rolf.simoes@@inpe.br}
#' @author Felipe Souza, \email{lipecaso@@gmail.com}
#' @author Alber Sanchez, \email{alber.ipia@@inpe.br}
#'
#' @description
#' R implementation of the QHAdam optimizer proposed
#' by Ma and Yarats(2019). We used the implementation available at
#' https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/qhadam.py.
#' Thanks to Nikolay Novik for providing the pytorch code.
#'
#' The original implementation has been developed by Facebook AI
#' and is licensed using the MIT license.
#'
#' From the the paper by Ma and Yarats(2019):
#' QHAdam is a QH augmented version of Adam, where we
#' replace both of Adam's moment estimators with quasi-hyperbolic terms.
#' QHAdam decouples the momentum term from the current gradient when
#' updating the weights, and decouples the mean squared gradients
#' term from the current squared gradient when updating the weights.
#'
#'
#' @references
#' Jerry Ma, Denis Yarats,
#' "Quasi-hyperbolic momentum and Adam for deep learning".
#'  https://arxiv.org/abs/1810.06801
#'
#' @param params         List of parameters to optimize.
#' @param lr             Learning rate (default: 1e-3)
#' @param betas          Coefficients computing running averages of gradient
#'                       and its square (default: (0.9, 0.999))
#' @param nus            Immediate discount factors used to
#'                       estimate the gradient and its square
#'                       (default: (1.0, 1.0))
#' @param eps            Term added to the denominator to improve numerical
#'                       stability (default: 1e-8)
#' @param weight_decay   Weight decay (L2 penalty) (default: 0)
#' @param decouple_weight_decay Whether to decouple the weight
#'                        decay from the gradient-based optimization step.
#'
#' @returns
#' A torch optimizer object implementing the `step` method.
#' @examples
#' if (torch::torch_is_installed()) {

#' # function to demonstrate optimization
#' beale <- function(x, y) {
#'     log((1.5 - x + x * y)^2 + (2.25 - x - x * y^2)^2 + (2.625 - x + x * y^3)^2)
#'  }
#' # define optimizer
#' optim <- torchopt::optim_qhadam
#' # define hyperparams
#' opt_hparams <- list(lr = 0.01)
#'
#' # starting point
#' x0 <- 3
#' y0 <- 3
#' # create tensor
#' x <- torch::torch_tensor(x0, requires_grad = TRUE)
#' y <- torch::torch_tensor(y0, requires_grad = TRUE)
#' # instantiate optimizer
#' optim <- do.call(optim, c(list(params = list(x, y)), opt_hparams))
#' # run optimizer
#' steps <- 400
#' x_steps <- numeric(steps)
#' y_steps <- numeric(steps)
#' for (i in seq_len(steps)) {
#'     x_steps[i] <- as.numeric(x)
#'     y_steps[i] <- as.numeric(y)
#'     optim$zero_grad()
#'     z <- beale(x, y)
#'     z$backward()
#'     optim$step()
#' }
#' print(paste0("starting value = ", beale(x0, y0)))
#' print(paste0("final value = ", beale(x_steps[steps], y_steps[steps])))
#' }
#'
#' @export
optim_qhadam <- torch::optimizer(
    classname = "optim_qhadam",
    initialize = function(params,
                          lr = 0.01,
                          betas = c(0.9, 0.999),
                          eps = 0.001,
                          nus = c(1.0, 1.0),
                          weight_decay = 0,
                          decouple_weight_decay = FALSE) {
        if (lr <= 0.0)
            stop("Learning rate must be positive.", call. = FALSE)
        if (eps < 0.0)
            stop("eps must be non-negative.", call. = FALSE)
        if (betas[1] > 1.0 | betas[1] <= 0.0)
            stop("Invalid beta parameter.", call. = FALSE)
        if (betas[2] > 1.0 | betas[1] <= 0.0)
            stop("Invalid beta parameter.", call. = FALSE)
        if (weight_decay < 0)
            stop("Invalid weight_decay value.", call. = FALSE)

        defaults = list(
            lr                    = lr,
            betas                 = betas,
            eps                   = eps,
            nus                   = nus,
            weight_decay          = weight_decay,
            decouple_weight_decay = decouple_weight_decay
        )
        super$initialize(params, defaults)
    },
    step = function(closure = NULL) {
        loop_fun <- function(group, param, g, p) {
            if (is.null(param$grad))
                next

            # define parameters
            beta1        <- group[['betas']][[1]]
            beta2        <- group[['betas']][[2]]
            nu1          <- group[['nus']][[1]]
            nu2          <- group[['nus']][[2]]
            weight_decay <- group[['weight_decay']]
            decouple_weight_decay <-  group[["decouple_weight_decay"]]
            eps          <- group[["eps"]]
            lr           <- group[['lr']]

            d_p <- param$grad

            if (weight_decay != 0) {
                if (decouple_weight_decay)
                    param$mul_(1 - lr * weight_decay)
                else
                    d_p$add_(weight_decay, p.data)
            }

            d_p_sq = d_p$mul(d_p)


            # State initialization
            # State initialization
            if (length(state(param)) == 0) {
                state(param) <- list()

                state(param)[["beta1_weight"]] <-  0.0
                state(param)[["beta2_weight"]] <-  0.0
                # Exponential moving average of gradient values
                state(param)[["exp_avg"]] <- torch::torch_zeros_like(param)
                # Exponential moving average of squared gradient values
                state(param)[["exp_avg_sq"]] <- torch::torch_zeros_like(param)
            }
            # Define variables for optimization function
            state(param)[["beta1_weight"]] <-  1.0 + beta1 * state(param)[["beta1_weight"]]
            state(param)[["beta2_weight"]] <-  1.0 + beta2 * state(param)[["beta2_weight"]]

            beta1_weight <-  state(param)[["beta1_weight"]]
            beta2_weight <-  state(param)[["beta2_weight"]]

            exp_avg      <- state(param)[["exp_avg"]]
            exp_avg_sq   <- state(param)[["exp_avg_sq"]]

            beta1_adj <-  1.0 - (1.0 / beta1_weight)
            beta2_adj <-  1.0 - (1.0 / beta2_weight)
            exp_avg$mul_(beta1_adj)$add_(d_p, alpha = 1.0 - beta1_adj)
            exp_avg_sq$mul_(beta2_adj)$add_(d_p_sq, alpha = 1.0 - beta2_adj)

            avg_grad <-  exp_avg$mul(nu1)
            if (nu1 != 1.0)
                avg_grad$add_(d_p, alpha = 1.0 - nu1)

            avg_grad_rms = exp_avg_sq$mul(nu2)
            if (nu2 != 1.0)
                avg_grad_rms$add_(d_p_sq, alpha = 1.0 - nu2)
            avg_grad_rms$sqrt_()
            if (eps != 0.0)
                avg_grad_rms$add_(eps)

            param$addcdiv_(avg_grad, avg_grad_rms, value = -lr)
        }
        private$step_helper(closure, loop_fun)
    }
)
