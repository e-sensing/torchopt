#' @title Adabelief optimizer
#'
#' @name optim_adabelief
#'
#' @author Gilberto Camara, \email{gilberto.camara@@inpe.br}
#' @author Rolf Simoes, \email{rolf.simoes@@inpe.br}
#' @author Felipe Souza, \email{lipecaso@@gmail.com}
#' @author Alber Sanchez, \email{alber.ipia@@inpe.br}
#'
#' @description
#' R implementation of the adabelief optimizer proposed
#' by Zhuang et al (2020). We used the pytorch implementation
#' developed by the authors which is available at
#' https://github.com/jettify/pytorch-optimizer.
#' Thanks to Nikolay Novik of his work on python optimizers.
#'
#' The original implementation is licensed using the Apache-2.0 software license.
#' This implementation is also licensed using Apache-2.0 license.
#'
#' From the abstract by the paper by Zhuang et al (2021):
#' We propose Adabelief to simultaneously achieve three goals:
#' fast convergence as in adaptive methods, good generalization as in SGD,
#' and training stability. The intuition for AdaBelief is to adapt
#' the stepsize according to the "belief" in the current gradient direction.
#' Viewing the exponential moving average of the noisy gradient
#' as the prediction of the gradient at the next time step,
#' if the observed gradient greatly deviates from the prediction,
#' we distrust the current observation and take a small step;
#' if the observed gradient is close to the prediction,
#' we trust it and take a large step.

#' @references
#' Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar Tatikonda,
#' Nicha Dvornek, Xenophon Papademetris, James S. Duncan.
#' "Adabelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients",
#' 34th Conference on Neural Information Processing Systems (NeurIPS 2020),
#' Vancouver, Canada.
#' https://arxiv.org/abs/2010.07468
#'
#' @param params            List of parameters to optimize.
#' @param lr                Learning rate (default: 1e-3)
#' @param betas             Coefficients for computing running averages
#'                          of gradient and its square (default: (0.9, 0.999))
#' @param eps               Term added to the denominator to improve numerical
#'                          stability (default: 1e-16)
#' @param weight_decay      Weight decay (L2 penalty) (default: 0)
#' @param weight_decouple   Use decoupled weight decay as is done in AdamW?
#' @param fixed_decay       This is used when weight_decouple is set as True.
#'                          When fixed_decay == True, weight decay is
#'                          W_new = W_old - W_old * decay.
#'                          When fixed_decay == False, the weight decay is
#'                          W_new = W_old - W_old * decay * learning_rate.
#'                          In this case, weight decay decreases with learning rate.
#' @param rectify           Perform the rectified update similar to RAdam?
#'
#' @returns
#' A torch optimizer object implementing the `step` method.
#'
#' @export
optim_adabelief <- torch::optimizer(
    classname = "optim_adabelief",
    initialize = function(params,
                          lr = 0.001,
                          betas = c(0.9, 0.999),
                          eps = 1.0e-08,
                          weight_decay = 1.0e-06,
                          weight_decouple = TRUE,
                          fixed_decay = FALSE,
                          rectify = TRUE) {
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

        self$weight_decouple <-  weight_decouple
        self$rectify <-  rectify
        self$fixed_decay <-  fixed_decay
    },
    step = function(closure = NULL){
        loop_fun <- function(group, param, g, p) {
            if (is.null(param$grad))
                next
            grad <- param$grad

            # Variable initialization
            beta1        <- group[['betas']][[1]]
            beta2        <- group[['betas']][[2]]
            weight_decay <- group[['weight_decay']]
            eps          <- group[["eps"]]
            lr           <- group[['lr']]

            # State initialization
            if (length(state(param)) == 0) {
                state(param) <- list()
                state(param)[["rho_inf"]] <- 2.0 / (1.0 - beta2) - 1.0
                state(param)[["step"]] <- 0
                # Exponential moving average of gradient values
                state(param)[["exp_avg"]] <- torch::torch_zeros_like(param)
                # Exponential moving average of squared gradient values
                state(param)[["exp_avg_var"]] <- torch::torch_zeros_like(param)
            }
            # Define variables for optimization function
            exp_avg      <- state(param)[["exp_avg"]]
            exp_avg_var  <- state(param)[["exp_avg_var"]]


            # take one step
            state(param)[["step"]] <- state(param)[["step"]] + 1
            # bias correction
            bias_correction1 <- 1 - beta1^state(param)[['step']]
            bias_correction2 <- 1 - beta2^state(param)[['step']]

            # perform weight decay, check if decoupled weight decay
            if (self$weight_decouple) {
                if (!self$fixed_decay)
                    param$mul_(1.0 - lr * weight_decay)
                else
                    param$mul_(1.0 - weight_decay)
            } else {
                if (weight_decay != 0)
                    grad$add_(param, alpha = weight_decay)
            }
            # update the first moment
            exp_avg$mul_(beta1)$add_(grad, alpha = 1 - beta1)
            grad_residual <- grad - exp_avg
            # Decay the second moment
            exp_avg_var$mul_(beta2)$addcmul_(grad_residual,
                                             grad_residual,
                                             value = (1 - beta2))

            # calculate denominator
            denom <- (exp_avg_var$add_(eps)$sqrt()/sqrt(bias_correction2))$add_(eps)

            if (!self$rectify) {
                # calculate step size
                step_size <- lr / bias_correction1
                param$addcdiv_(exp_avg, denom, value = -step_size)
            } else {
                # calculate rho_t
                rho_inf <- state(param)[["rho_inf"]]
                step <- state(param)[["step"]]
                state(param)[["rho_t"]] <- rho_inf  -
                    (2 * step * beta2 ^ step) /
                    (1.0 - beta2 ^ step)
                rho_t <- state(param)[["rho_t"]]

                # more conservative since it's an approximated value
                if (rho_t > 4) {
                    # perform Adam style update if variance is small
                    rt = (
                        (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                        / (rho_inf - 4.0)
                        / (rho_inf - 2.0)
                        / rho_t
                    )
                    rt = sqrt(rt)
                    step_size <- rt * lr / bias_correction1
                    param$addcdiv_(exp_avg,
                                   denom,
                                   value = -step_size
                    )
                } else
                    # perform SGD style update
                    param$add_(exp_avg, alpha = -lr)
            }
        }
        private$step_helper(closure, loop_fun)
    }
)
