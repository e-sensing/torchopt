#' @title AdamW optimizer
#'
#' @name optim_radam
#'
#' @author Gilberto Camara, \email{gilberto.camara@@inpe.br}
#' @author Daniel Falbel, \email{daniel.falble@@gmail.com}
#' @author Rolf Simoes, \email{rolf.simoes@@inpe.br}
#' @author Felipe Souza, \email{lipecaso@@gmail.com}
#' @author Alber Sanchez, \email{alber.ipia@@inpe.br}
#'
#' @description
#' R implementation of the RAdam optimizer proposed
#' by Liu et al. (2019).
#' We used the implementation in PyTorch as a basis for our
#' implementation.
#'
#' From the abstract by the paper by Liu et al. (2019):
#' The learning rate warmup heuristic achieves remarkable success
#' in stabilizing training, accelerating convergence and improving
#' generalization for adaptive stochastic optimization algorithms
#' like RMSprop and Adam. Here, we study its mechanism in details.
#' Pursuing the theory behind warmup, we identify a problem of the
#' adaptive learning rate (i.e., it has problematically large variance
#' in the early stage), suggest warmup works as a variance reduction
#' technique, and provide both empirical and theoretical evidence to verify
#' our hypothesis. We further propose RAdam, a new variant of Adam,
#' by introducing a term to rectify the variance of the adaptive learning rate.
#' Extensive experimental results on image classification, language modeling,
#' and neural machine translation verify our intuition and demonstrate
#' the effectiveness and robustness of our proposed method.
#'
#' @references
#' Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen,
#' Xiaodong Liu, Jianfeng Gao, Jiawei Han,
#' "On the Variance of the Adaptive Learning Rate and Beyond",
#' International Conference on Learning Representations (ICLR) 2020.
#' https://arxiv.org/abs/1908.03265
#'
#' @param params       List of parameters to optimize.
#' @param lr           Learning rate (default: 1e-3)
#' @param betas        Coefficients computing running averages of gradient
#'   and its square (default: (0.9, 0.999))
#' @param eps          Term added to the denominator to improve numerical
#'   stability (default: 1e-8)
#' @param weight_decay Weight decay (L2 penalty) (default: 0)
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
#' optim <- torchopt::optim_radam
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
optim_radam <- torch::optimizer(
    "optim_radam",
    initialize = function(params,
                          lr = 0.01,
                          betas = c(0.9, 0.999),
                          eps = 1e-8,
                          weight_decay = 0) {
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
            lr           = lr,
            betas        = betas,
            eps          = eps,
            weight_decay = weight_decay
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
                state(param)[["step"]] <- 0
                # Exponential moving average of gradient values
                state(param)[["exp_avg"]] <- torch::torch_zeros_like(param)
                # Exponential moving average of squared gradient values
                state(param)[["exp_avg_sq"]] <- torch::torch_zeros_like(param)
            }
            # Define variables for optimization function
            exp_avg      <- state(param)[["exp_avg"]]
            exp_avg_sq   <- state(param)[["exp_avg_sq"]]
            beta1        <- group[['betas']][[1]]
            beta2        <- group[['betas']][[2]]
            weight_decay <- group[['weight_decay']]
            eps          <- group[["eps"]]
            lr           <- group[['lr']]

            # take one step
            state(param)[["step"]] <- state(param)[["step"]] + 1
            step <- state(param)[["step"]]

            # bias correction
            bias_correction1 <- 1 - beta1^state(param)[['step']]
            bias_correction2 <- 1 - beta2^state(param)[['step']]

            # L2 correction
            if (weight_decay != 0)
                grad$add_(param, alpha = weight_decay)


            # Decay the first moment
            exp_avg$mul_(beta1)$add_(grad, alpha = 1 - beta1)
            # Decay the second moment
            exp_avg_sq$mul_(beta2)$addcmul_(grad, grad, value = (1 - beta2))

            # correcting bias for the first moving moment
            bias_corrected_exp_avg <-  exp_avg / bias_correction1

            # maximum length of the approximated SMA
            rho_inf <-  2 / (1 - beta2) - 1
            # compute the length of the approximated SMA
            rho_t <-  rho_inf - 2 * step * (beta2^step) / bias_correction2
            # adjust learning rate
            if (rho_t > 5.0) {
                # Compute the variance rectification term and update parameters accordingly
                rect <- sqrt((rho_t - 4) * (rho_t - 2) * rho_inf /
                                 ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                adaptive_lr <- sqrt(bias_correction2) / exp_avg_sq$sqrt()$add_(eps)
                param$add_(bias_corrected_exp_avg * lr * adaptive_lr * rect, alpha = -1.0)
            } else
                param$add_(bias_corrected_exp_avg * lr, alpha =- 1.0)
        }
        private$step_helper(closure, loop_fun)
    }
)
