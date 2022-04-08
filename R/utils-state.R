# get state attribute
state <- function(self) {
    attr(self, "state")
}

# set state attribute
`state<-` <- function(self, value) {
    attr(self, "state") <- value
    self
}
