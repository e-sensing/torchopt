#' @title SWATS optimizer
#'
#' @name optim_swats
#'
#' @author Gilberto Camara, \email{gilberto.camara@@inpe.br}
#' @author Daniel Falbel, \email{daniel.falble@@gmail.com}
#' @author Rolf Simoes, \email{rolf.simoes@@inpe.br}
#' @author Felipe Souza, \email{lipecaso@@gmail.com}
#' @author Alber Sanchez, \email{alber.ipia@@inpe.br}
#'
#' @description
#' R implementation of the SWATS optimizer proposed
#' by Shekar and Sochee (2018).
#' We used the implementation available at
#' https://github.com/jettify/pytorch-optimizer/
#' Thanks to Nikolay Novik for providing the pytorch code.
#'
#' From the abstract by the paper by Shekar and Sochee (2018):
#' Adaptive optimization methods such as Adam, Adagrad or RMSprop
#' have been found to generalize poorly compared to
#' Stochastic gradient descent (SGD). These methods tend to perform well i
#' in the initial portion of training but are outperformed by SGD at
#' later stages of training. We investigate a hybrid strategy that begins
#' training with an adaptive method and switches to SGD
#' when a triggering condition is satisfied.
#' The condition we propose relates to the projection of Adam
#' steps on the gradient subspace. By design, the monitoring process
#' for this condition adds very little overhead and does not increase
#' the number of hyperparameters in the optimizer.
#'
#' @references
#' Nitish Shirish Keskar, Richard Socher
#' "Improving Generalization Performance by Switching from Adam to SGD".
#' International Conference on Learning Representations (ICLR) 2018.
#' https://arxiv.org/abs/1712.07628
#'
#' @param params       List of parameters to optimize.
#' @param lr           Learning rate (default: 1e-3)
#' @param betas        Coefficients computing running averages of gradient
#'                     and its square (default: (0.9, 0.999)).
#' @param eps          Term added to the denominator to improve numerical
#'                     stability (default: 1e-8).
#' @param weight_decay Weight decay (L2 penalty) (default: 0).
#' @param nesterov     Enables Nesterov momentum (default: False).
#'
#' @returns
#' A torch optimizer object implementing the `step` method.
#'
#' @export
optim_swats <- torch::optimizer(
    classname = "optim_swats",
    initialize = function(params,
                          lr = 0.01,
                          betas = c(0.9, 0.999),
                          eps = 1e-8,
                          weight_decay = 0,
                          nesterov = FALSE) {
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
            weight_decay = weight_decay,
            nesterov     = nesterov,
            phase        = "ADAM"
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
                # create momentum buffer
                state(param)[["momentum_buffer"]] <- NA
                # Exponential moving average of gradient values
                state(param)[["exp_avg"]] <- torch::torch_zeros_like(param)
                # Exponential moving average of squared gradient values
                state(param)[["exp_avg_sq"]] <- torch::torch_zeros_like(param)
                # moving average for the non-orthogonal projection scaling
                # state(param)[["exp_avg2"]] <- param$new(1)$fill_(0)
                state(param)[["exp_avg2"]] <- param$new_zeros(1)
            }
            # Define variables for optimization function
            exp_avg      <- state(param)[["exp_avg"]]
            exp_avg_sq   <- state(param)[["exp_avg_sq"]]
            exp_avg2     <- state(param)[["exp_avg2"]]
            beta1        <- group[['betas']][[1]]
            beta2        <- group[['betas']][[2]]
            weight_decay <- group[['weight_decay']]
            eps          <- group[["eps"]]
            lr           <- group[['lr']]
            phase        <- group[["phase"]]
            nesterov     <- group[["nesterov"]]

            # take one step
            state(param)[["step"]] <- state(param)[["step"]] + 1
            step <- state(param)[["step"]]

            # L2 correction
            if (weight_decay != 0)
                grad$add_(param, alpha = weight_decay)

            # if its SGD phase, take an SGD update and continue
            if (phase == 'SGD'){
                if (is.na(state(param)[["momentum_buffer"]])) {
                    state(param)[["momentum_buffer"]] <-
                        torch::torch_clone(grad)$detach()
                    buf <- state(param)[["momentum_buffer"]]
                } else {
                    buf <- state(param)[["momentum_buffer"]]
                    buf$mul_(beta1)$add_(grad)
                    grad <-  buf
                    grad$mul_(1 - beta1)
                    if (nesterov)
                        grad$add_(buf, alpha = beta1)
                    param$add_(grad, alpha = -lr)
                    next
                }
            }

            # Decay the first moment
            exp_avg$mul_(beta1)$add_(grad, alpha = 1 - beta1)
            # Decay the second moment
            exp_avg_sq$mul_(beta2)$addcmul_(grad, grad, value = (1 - beta2))
            # calculate denominator
            denom = exp_avg_sq$sqrt()$add_(eps)

            # bias correction
            bias_correction1 <- 1 - beta1^state(param)[['step']]
            bias_correction2 <- 1 - beta2^state(param)[['step']]

            # calculate step size
            step_size <- lr * (bias_correction2 ^ 0.5) / bias_correction1

            pf <-  -step_size * (exp_avg / denom)
            param$add_(pf)

            p_view <-  pf$view(-1)
            pg <- p_view$dot(grad$view(-1))

            if (as.logical(pg != 0)) {
                # the non-orthognal scaling estimate
                scaling <-  p_view$dot(p_view) / -pg
                exp_avg2$mul_(beta2)$add_(scaling, alpha = (1 - beta2))

                # bias corrected exponential average
                corrected_exp_avg <- exp_avg2 / bias_correction2

                # checking criteria of switching to SGD training
                if (as.logical(state(param)[['step']] > 1) &&
                    as.logical(corrected_exp_avg$allclose(scaling, rtol = 1e-6)) &&
                    as.logical(corrected_exp_avg > 0)
                ) {
                    group[['phase']] <-  'SGD'
                    group[['lr']] <- corrected_exp_avg$item()
                }
            }
        }
        private$step_helper(closure, loop_fun)
    }
)
