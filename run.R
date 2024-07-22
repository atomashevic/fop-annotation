install_if_missing <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package)
  }
}

# Install devtools, shinyjs, and shinydashboard if not already installed
install_if_missing("devtools")
install_if_missing("shinyjs")
install_if_missing("shinydashboard")

# Load devtools package
library(devtools)

# Install local package 'taipan' if not already installed
if (!requireNamespace("taipan", quietly = TRUE)) {
  install_local("taipan_0.1.2.tar.gz", force = TRUE, upgrade = "never")
}

library(taipan)
library(shiny)
library(shinydashboard)


data <- read.csv("task.csv")

questions <- taipanQuestions(
  scene = div(
    radioButtons(
      "face",
      label = "Can you cleraly see a face of a politician on this image?",
      choices = c("Yes", "No"),
      selected = character(0)
    ),
    radioButtons(
      "label",
      label = "Is this face expressing positive emotion,
      negative emotion or is it a neutral expression?",
      choices = c(
        "Positive Emotion",
        "Negative Emotion",
        "Neutral Expression"
      ),
      selected = character(0)
    )
  ),
  selection = div()
)

buildTaipan(
  questions = questions,
  images = data$new_image_path,
  appdir = file.path("app/"),
  overwrite = TRUE,
  skip_check = TRUE
)
