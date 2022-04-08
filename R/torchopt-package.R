#' @keywords internal
"_PACKAGE"

# Include the following global variables
utils::globalVariables(c("self", "super", "ctx", "private"))

# Setup config options
.env <- new.env()

assign(x = "bg_xy_breaks", value = 255, envir = .env)
assign(x = "bg_z_breaks", value = 255, envir = .env)
assign(x = "bg_palette", value = "viridis", envir = .env)
assign(x = "bg_contour_levels", value = 10, envir = .env)
assign(x = "bg_contour_labels", value = FALSE, envir = .env)
assign(x = "bg_contour_color", value = "#FFFFFF50", envir = .env)

