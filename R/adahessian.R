#'@title Adahessian optimizer
#'
#'@name optim_adahessian
#'
#'@author Rolf Simoes, \email{rolf.simoes@@inpe.br}
#'@author Felipe Souza, \email{lipecaso@@gmail.com}
#'@author Alber Sanchez, \email{alber.ipia@@inpe.br}
#'@author Gilberto Camara, \email{gilberto.camara@@inpe.br}
#'
#'@description  R implementation of the Adahessian optimizer proposed
#' by Yao et al.(2020). The original implementation is available at
#' https://github.com/amirgholami/adahessian.
#'
#' @references
#' Yao, Z., Gholami, A., Shen, S., Mustafa, M., Keutzer, K.,
#' & Mahoney, M. (2021).
#' ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning.
#' Proceedings of the AAAI Conference on Artificial Intelligence, 35(12),
#' 10665-10673.
#' https://arxiv.org/abs/2006.00719
#'
#' @param params                        Iterable of parameters to optimize.
#' @param lr                            Learning rate (default: 0.15).
#' @param betas                         Coefficients for computing
#'                                      running averages of gradient
#'                                      and is square(default: (0.9, 0.999)).
#' @param eps                           Term added to the denominator to improve
#'                                      numerical stability (default: 1e-4).
#' @param weight_decay                  L2 penalty (default: 0).
#' @param hessian_power                 Hessian power (default: 1.0).
#'
#'
#' @returns
#' An optimizer object implementing the `step` and `zero_grad` methods.
#' @examples
#' if (torch::torch_is_installed()) {

#' # function to demonstrate optimization
#' beale <- function(x, y) {
#'     log((1.5 - x + x * y)^2 + (2.25 - x - x * y^2)^2 + (2.625 - x + x * y^3)^2)
#'  }
#' # define optimizer
#' optim <- optim_adahessian
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
#'     z$backward(retain_graph = TRUE, create_graph = TRUE)
#'     optim$step()
#' }
#' print(paste0("starting value = ", beale(x0, y0)))
#' print(paste0("final value = ", beale(x_steps[steps], y_steps[steps])))
#' }
#' @export
optim_adahessian <- torch::optimizer(
    classname = "optim_adahessian",
    initialize = function(
        params,
        lr = 0.15,
        betas = c(0.9, 0.999),
        eps = 1e-4,
        weight_decay = 0,
        hessian_power = 0.5
    ) {
        if (lr <= 0.0)
            rlang::abort("Learning rate must be positive.")
        if (eps <= 0.0)
            rlang::abort("eps must be non-negative.")
        if (betas[1] > 1.0 | betas[1] <= 0.0)
            rlang::abort("Invalid beta1 parameter.")
        if (betas[2] > 1.0 | betas[2] <= 0.0)
            rlang::abort("Invalid beta2 parameter.")
        if (hessian_power > 1.0 | hessian_power <= 0.0)
            rlang::abort("Invalid hessian power parameter.")
        if (weight_decay < 0)
            rlang::abort("Invalid weight_decay value")

        torch::torch_manual_seed(sample.int(10^5, 1))

        defaults = list(
            lr           = lr,
            betas        = betas,
            eps          = eps,
            hessian_power = hessian_power,
            weight_decay = weight_decay
        )
        super$initialize(params, defaults)
    },
    #     Get an estimate of Hessian Trace.
    #     This is done by computing the Hessian vector product with a random
    #     vector v at the current gradient point, to estimate Hessian trace by
    #     computing the gradient of <gradsH,v>.
    get_trace = function(params, grads){
        # Check backward was called with create_graph set to True
        purrr::map(grads, function(g) {
            if (purrr::is_null(g$grad_fn)) {
                msg <- paste("Gradient tensor does not have grad_fn",
                             "When calling loss.backward(), set create_graph to True.")
                rlang::abort(msg)
            }
        })
        # list of random tensors [-1, 1] to estimate Hessian matrix diagonal
        v <- purrr::map(params, function(p){
            return(2 * torch::torch_randint_like(input = p,
                                                 low = 0,
                                                 high = 2) - 1)
        })
        # Computes the sum of gradients of outputs w.r.t. the inputs.
        hvs <- torch::autograd_grad(
            outputs = grads,
            inputs  = params,
            grad_outputs = v,
            retain_graph = TRUE,
            create_graph = TRUE
        )

        # calculate hutchinson trace
        # approximation of hessian diagonal
        hutchinson_trace <- purrr::map(seq_along(hvs), function(hv_ind){
            hv <- hvs[[hv_ind]]
            param_size <-  hv$size()
            hv_abs <- hv$abs()
            if (length(param_size) <= 2) {
                return(hv_abs)
            } else if (length(param_size) == 3) {
                return(torch::torch_mean(hv_abs, dim = 1, keepdim = TRUE))
            } else if (length(param_size) == 4) {
                return(torch::torch_mean(hv_abs, dim = c(2, 3), keepdim = TRUE))
            } else
                rlang::abort("Only 1D to 4D tensors are supported.")
        })
        return(hutchinson_trace)
    },
    step = function(closure = NULL) {

        # #  Flatten params and grads into lists
        groups <- self$param_groups[[1]]
        params <- purrr::map(groups$params, function(pg){
                return(pg)
        })
        grads <- purrr::map(params, function(p) {
            if (!is.null(p$grad))
                return(p$grad)
        })
        # Get the Hessian diagonal
        self$hut_traces <- self$get_trace(params, grads)

        loop_fun <- function(group, param, g, p) {

            # state initialization
            if (length(state(param)) == 0) {
                state(param) <- list()
                state(param)[["step"]] <- 0
                state(param)[["exp_avg"]] <- torch::torch_zeros_like(param)
                state(param)[["exp_hessian_diag_sq"]] <- torch::torch_zeros_like(param)
            }
            # Perform correct stepweight decay as in AdamW
            # param$mul_(1 - group[['lr']] * group[['weight_decay']])

            exp_avg             <- state(param)[["exp_avg"]]
            exp_hessian_diag_sq <- state(param)[["exp_hessian_diag_sq"]]

            # increase step
            state(param)[["step"]] <- state(param)[["step"]] + 1

            # parameters for optimizer
            beta1 <-  group[['betas']][[1]]
            beta2 <-  group[['betas']][[2]]
            lr    <-  group[['lr']]
            eps   <-  group[['eps']]
            wd    <-  group[['weight_decay']]
            k     <-  group[['hessian_power']]
            step  <-  state(param)[["step"]]


            # Decay the first and second moment
            # running average coefficient
            exp_avg$mul_(beta1)$add_(param$grad, alpha = 1 - beta1)
            exp_hessian_diag_sq$mul_(beta2)$addcmul_(
                self$hut_traces[[p]],
                self$hut_traces[[p]],
                value = 1 - beta2
            )

            # bias correction
            bias_correction1 <-  1 - beta1 ^ step
            bias_correction2 <-  1 - beta2 ^ step
            sqrt_bc2 <- sqrt(bias_correction2)


            # make the square root, and the Hessian power
            denom <- ((exp_hessian_diag_sq$sqrt() ^ k) / (sqrt_bc2 ^ k))$add_(eps)

            # update
            param$sub_(lr * (exp_avg / bias_correction1 / denom
                             + wd * param))
        }
        private$step_helper(closure, loop_fun)
    }
)
