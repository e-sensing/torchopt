#' @title MADGRAD optimizer
#'
#' @name optim_madgrad
#' @rdname optim_madgrad
#' @importFrom madgrad optim_madgrad
#'
#' @author Daniel Falbel, \email{dfalbel@@gmail.com}
#'
#' @description
#' A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
#' Optimization (MADGRAD) is a general purpose optimizer that
#' can be used in place of SGD or Adam may converge faster and generalize
#' better. Currently GPU-only. Typically, the same learning rate schedule
#' that is used for SGD or Adam may be used. The overall learning rate is
#' not comparable to either method and should be determined by a
#' hyper-parameter sweep.
#'
#' MADGRAD requires less weight decay than other methods, often as little as
#' zero. Momentum values used for SGD or Adam's beta1 should work here also.
#'
#' On sparse problems both weight_decay and momentum should be set to 0.
#' (not yet supported in the R implementation).
#'
#' See \code{madgrad::\link[madgrad:optim_madgrad]{optim_madgrad}} for details.
#'
#' @references
#' Aaron Defazio, Samy Jelassi,
#' "Adaptivity without Compromise: A Momentumized, Adaptive, Dual
#' Averaged Gradient Method for Stochastic Optimization",
#' arXiv preprint arXiv:2101.11075, 2021.
#' https://doi.org/10.48550/arXiv.2101.11075
#'
#' @param params        List of parameters to optimize.
#' @param lr            Learning rate (default: 1e-2).
#' @param momentum      Momentum value in  the range [0,1) (default: 0.9).
#' @param weight_decay  Weight decay, i.e. a L2 penalty (default: 0).
#' @param eps           Term added to the denominator outside of
#'                      the root operation to improve numerical stability
#'                      (default: 1e-6).
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' library(torch)
#' x <- torch_randn(1, requires_grad = TRUE)
#' opt <- optim_madgrad(x)
#' for (i in 1:100) {
#'   opt$zero_grad()
#'   y <- x^2
#'   y$backward()
#'   opt$step()
#' }
#' all.equal(x$item(), 0, tolerance = 1e-9)
#' }
#'
#' @returns
#' A torch optimizer object implementing the `step` method.
#'
#' @export
NULL
