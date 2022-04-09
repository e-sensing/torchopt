#' @title Imported function
#' @author Daniel Falbel, \email{dfalbel@@gmail.com}
#' @keywords internal
#' @description Code lifted from a internal function of madgrad package.
#'   Get 'state' attribute of an object.
state <- function(self) {
    attr(self, "state")
}

#' @title Imported function
#' @author Daniel Falbel, \email{dfalbel@@gmail.com}
#' @keywords internal
#' @description Code lifted from a internal function of madgrad package.
#'   Set 'state' attribute of an object.
`state<-` <- function(self, value) {
    attr(self, "state") <- value
    self
}
