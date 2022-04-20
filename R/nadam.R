#' @title Nadam optimizer
#'
#' @name optim_nadam
#'
#' @author Gilberto Camara, \email{gilberto.camara@@inpe.br}
#' @author Rolf Simoes, \email{rolf.simoes@@inpe.br}
#' @author Felipe Souza, \email{lipecaso@@gmail.com}
#' @author Alber Sanchez, \email{alber.ipia@@inpe.br}
#'
#' @description
#' R implementation of the Nadam optimizer proposed
#' by Dazat (2016).
#'
#' From the abstract by the paper by Dozat (2016):
#' This work aims to improve upon the recently proposed and
#' rapidly popularized optimization algorithm Adam (Kingma & Ba, 2014).
#' Adam has two main components—a momentum component and an adaptive
#' learning rate component. However, regular momentum can be shown conceptually
#' and empirically to be inferior to a similar algorithm known as
#' Nesterov’s accelerated gradient (NAG).
#'
#' @references
#' Timothy Dozat,
#' "Incorporating Nesterov Momentum into Adam",
#' International Conference on Learning Representations (ICLR) 2016.
#' https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf
#'
#' @param params              List of parameters to optimize.
#' @param lr                  Learning rate (default: 1e-3)
#' @param betas               Coefficients computing running averages of gradient
#'                            and its square (default: (0.9, 0.999)).
#' @param eps                 Term added to the denominator to improve numerical
#'                            stability (default: 1e-8).
#' @param weight_decay        Weight decay (L2 penalty) (default: 0).
#' @param momentum_decay      Momentum_decay (default: 4e-3).
#'
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
#' optim <- torchopt::optim_nadam
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
#' @export
optim_nadam <- torch::optimizer(
    classname = "optim_nadam",
    initialize = function(params,
                          lr = 0.002,
                          betas = c(0.9, 0.999),
                          eps = 1e-8,
                          weight_decay = 0,
                          momentum_decay = 4.0e-03) {
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
        if (momentum_decay < 0)
            stop("Invalid momentum_decay value.", call. = FALSE)

        defaults = list(
            lr             = lr,
            betas          = betas,
            eps            = eps,
            weight_decay   = weight_decay,
            momentum_decay = momentum_decay
        )
        super$initialize(params, defaults)
    },
    step = function(closure = NULL){
        loop_fun <- function(group, param, g, p) {
            if (is.null(param$grad))
                next
            grad <- param$grad

            # State initialization
            if (length(state(param)) == 0) {
                state(param) <- list()
                state(param)[["step"]] <- torch::torch_tensor(0)
                # momentum product
                state(param)[["mu_product"]] <- torch::torch_tensor(1.)
                # Exponential moving average of gradient values
                state(param)[["exp_avg"]] <- torch::torch_zeros_like(param)
                # Exponential moving average of squared gradient values
                state(param)[["exp_avg_sq"]] <- torch::torch_zeros_like(param)
            }
            # Define variables for optimization function
            exp_avg      <- state(param)[["exp_avg"]]
            exp_avg_sq   <- state(param)[["exp_avg_sq"]]
            step         <- state(param)[["step"]]
            mu_product   <- state(param)[["mu_product"]]
            beta1        <- group[['betas']][[1]]
            beta2        <- group[['betas']][[2]]
            weight_decay <- group[['weight_decay']]
            eps          <- group[["eps"]]
            lr           <- group[['lr']]
            momentum_decay <- group[["momentum_decay"]]

            # take one step
            state(param)[["step"]] <- state(param)[["step"]] + 1

            # bias correction
            bias_correction2 <- 1 - beta2^state(param)[['step']]

            # weight_decay
            if (weight_decay != 0)
               grad = grad$add(param, alpha = weight_decay)

            # calculate the momentum cache \mu^{t} and \mu^{t+1}
            mu = beta1 * (1. - 0.5 * (0.96 ^ (step * momentum_decay)))
            mu_next = beta1 * (1. - 0.5 * (0.96 ^ ((step + 1) * momentum_decay)))

            # update momentum
            mu_product <- mu_product * mu
            mu_product_next <- mu_product * mu * mu_next

            # decay the first and second moment running average coefficient
            exp_avg$mul_(beta1)$add_(grad, alpha = 1 - beta1)
            exp_avg_sq$mul_(beta2)$addcmul_(grad, grad, value = 1 - beta2)

            # calculate denominator
            denom = exp_avg_sq$div(bias_correction2)$sqrt()$add_(eps)

            # update objective function
            param$addcdiv_(grad, denom,
                           value = -lr * (1. - mu) / (1. - mu_product$item()))
            param$addcdiv_(exp_avg, denom,
                           value = -lr * mu_next / (1. - mu_product_next$item()))

        }
        private$step_helper(closure, loop_fun)
    }
)
