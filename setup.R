# NOTE: working directory is /fop-annotation
install_if_missing <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package)
  }
}

install_if_missing("devtools")
install_if_missing("shinyjs")
install_if_missing("shinydashboard")

if (!require("taipan")) {
  devtools::install_local("taipan_0.1.2.tar.gz",
    force = TRUE,
    upgrade = "never"
  )
}

library(taipan)
library(shiny)
library(shinydashboard)
library(dplyr)

generate_unique_name <- function() {
  paste0("frame_E_", sprintf("%04d", sample(1000:9999, 1)), ".jpg")
}

load("sample.Rds")

images <- sample

images <- paste0("fop/", images)

ids <- gsub("fop/", "", images)
ids <- gsub("frame_", "", ids)
ids <- gsub(".jpg", "", ids)

data <- data.frame(id = ids, image = images)

generate_unique_name <- function() {
  paste0("frame_E_", sprintf("%04d", sample(1000:9999, 1)), ".jpg")
}

output_dir <- "fop"

data <- data %>%
  rowwise() %>%
  mutate(new_image_path = {
    old_filepath <- image
    new_filename <- generate_unique_name()
    new_filepath <- file.path(output_dir, new_filename)
    file.copy(old_filepath, new_filepath)
    new_filepath
  }) %>%
  ungroup()

write.csv(data, "task.csv", row.names = FALSE)
