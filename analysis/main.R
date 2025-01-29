install_if_missing <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    install.packages(package)
  }
}

install_if_missing("tidyverse")

install_if_missing("irr")
install_if_missing("ggplot2")

library(tidyverse)
library(irr)
library(ggplot2)
library(dplyr)

# Function to preprocess human coder data
preprocess_human_data <- function(data) {
  data %>%
    mutate(label = ifelse(face == "No", "No face", label))
}

# Update the preprocess_ml_data function
preprocess_ml_data <- function(data, model_name) {
  if (model_name == "CNN-fer") {
    data %>%
      mutate(
        label = case_when(
          outcome == "PS" ~ "Positive Emotion",
          outcome == "NG" ~ "Negative Emotion",
          outcome == "NT" ~ "Neutral Expression",
          outcome == "NO" ~ "No face",
          TRUE ~ NA_character_
        )
      ) %>%
      select(image_name, label)
  } else {
    data %>%
      mutate(
        label = case_when(
          face == "No" ~ "No face",
          label == "Neutral Expression" ~ "Neutral Expression",
          label == "Positive Emotion" ~ "Positive Emotion",
          label == "Negative Emotion" ~ "Negative Emotion",
          TRUE ~ NA_character_
        )
      ) %>%
      select(image_name, label)
  }
}

calculate_f1_score <- function(confusion_matrix) {
  precision <- diag(confusion_matrix) / colSums(confusion_matrix)
  recall <- diag(confusion_matrix) / rowSums(confusion_matrix)
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(mean(f1, na.rm = TRUE))
}

save_plot <- function(plot, filename) {
  png(paste0("figures/main/", filename, ".png"), width = 800, height = 600)
  print(plot)
  dev.off()
}

# Add this function for calculating Cohen's Kappa
calculate_cohens_kappa <- function(rater1, rater2) {
  require(irr)
  kappa2(data.frame(rater1, rater2))$value
}

# Add this function for calculating Fleiss' Kappa
calculate_fleiss_kappa <- function(ratings) {
  require(irr)
  
  # Convert ratings to a matrix
  ratings_matrix <- as.matrix(ratings)
  
  # Calculate Fleiss' Kappa
  kappa_result <- kappam.fleiss(ratings_matrix)
  
  return(kappa_result$value)
}

# Function to print label summary
print_label_summary <- function(labels, name) {
  cat("\n", name, "label summary:\n")
  print(table(labels))
  cat("Unique labels:", paste(unique(labels), collapse = ", "), "\n")
}

# Read and preprocess the human coder CSV files
data1 <- read.csv("results/C1.csv") %>% preprocess_human_data()
data2 <- read.csv("results/C2.csv") %>% preprocess_human_data()
data3 <- read.csv("results/C3.csv") %>% preprocess_human_data()
data4 <- read.csv("results/C4.csv") %>% preprocess_human_data()
data5 <- read.csv("results/C5.csv") %>% preprocess_human_data()

# Update the list of ML models to include MetaCLIP
ml_models <- c("CNN-fer", "CNN-strong", "CNN-weak", "CLIP", "CLIP-NE", "MetaCLIP", "MetaCLIP-NE", "appleCLIP", "appleCLIP-NE")

# Read and preprocess the ML algorithm CSV files
ml_data <- list()
for (model in ml_models) {
  tryCatch({
    data <- read.csv(paste0("results/", model, ".csv"))
    ml_data[[model]] <- preprocess_ml_data(data, model_name = model)
    print(paste("Successfully processed", model))
  }, error = function(e) {
    warning(paste("Error processing", model, ":", e$message))
    ml_data[[model]] <- NULL
  })
}

# Remove any NULL entries from ml_data
ml_data <- ml_data[!sapply(ml_data, is.null)]
ml_models <- names(ml_data)

# Combine the data into a single dataframe
combined_data <- data1 %>%
  select(image_name, label) %>%
  rename(rater1 = label) %>%
  left_join(data2 %>% select(image_name, label) %>% rename(rater2 = label), by = "image_name") %>%
  left_join(data3 %>% select(image_name, label) %>% rename(rater3 = label), by = "image_name") %>%
  left_join(data4 %>% select(image_name, label) %>% rename(rater4 = label), by = "image_name") %>%
  left_join(data5 %>% select(image_name, label) %>% rename(rater5 = label), by = "image_name")

# Add ML model predictions to the combined data
for (model in ml_models) {
  combined_data <- combined_data %>%
    left_join(ml_data[[model]] %>% rename(!!model := label), by = "image_name")
}

# Calculate human majority vote
combined_data <- combined_data %>%
  rowwise() %>%
  mutate(
    human_majority = {
      votes <- table(c(rater1, rater2, rater3, rater4, rater5))
      if (max(votes) >= 3) names(which.max(votes)) else NA_character_
    }
  ) %>%
  ungroup()

calculate_metrics <- function(true_labels, predicted_labels, all_human_labels) {
  # Filter out cases where there's no majority vote
  valid_cases <- !is.na(true_labels)
  true_labels <- true_labels[valid_cases]
  predicted_labels <- predicted_labels[valid_cases]
  all_human_labels <- all_human_labels[valid_cases, , drop = FALSE]
  
  # Ensure both sets have the same categories
  all_categories <- unique(c(true_labels, predicted_labels))
  
  # Create confusion matrix with all categories
  confusion_matrix <- table(factor(true_labels, levels = all_categories),
                            factor(predicted_labels, levels = all_categories))
  
  cat("\nConfusion Matrix:\n")
  print(confusion_matrix)
  
  # Calculate precision and recall for each category
  precision <- diag(confusion_matrix) / colSums(confusion_matrix)
  recall <- diag(confusion_matrix) / rowSums(confusion_matrix)
  
  # Calculate F1 score for each category
  f1 <- ifelse(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall))
  
  # Calculate macro F1
  macro_f1 <- mean(f1, na.rm = TRUE)
  
  # Calculate weighted F1
  weights <- table(true_labels) / length(true_labels)
  weighted_f1 <- sum(f1 * weights[names(f1)], na.rm = TRUE)
  
  # Calculate accuracy
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  
  # Calculate Matthews Correlation Coefficient (MCC)
  library(mltools)
  mcc <- mcc(preds = predicted_labels, actuals = true_labels)
  
  # Calculate Cohen's Kappa
  cohens_kappa <- calculate_cohens_kappa(true_labels, predicted_labels)
  
  # Calculate Fleiss' Kappa
  all_ratings <- cbind(all_human_labels, predicted_labels)
  fleiss_kappa <- calculate_fleiss_kappa(all_ratings)
  
  # Prepare results
  results <- list(
    precision = precision,
    recall = recall,
    macro_f1 = macro_f1,
    weighted_f1 = weighted_f1,
    accuracy = accuracy,
    mcc = mcc,
    cohens_kappa = cohens_kappa,
    fleiss_kappa = fleiss_kappa
  )
  
  return(results)
}

# Initialize metrics_results dataframe
metrics_results <- data.frame(model = character(), 
                              precision_positive = numeric(),
                              precision_negative = numeric(),
                              precision_neutral = numeric(),
                              precision_noface = numeric(),
                              recall_positive = numeric(),
                              recall_negative = numeric(),
                              recall_neutral = numeric(),
                              recall_noface = numeric(),
                              macro_f1 = numeric(),
                              weighted_f1 = numeric(),
                              accuracy = numeric(),
                              mcc = numeric(),
                              cohens_kappa = numeric(),
                              fleiss_kappa = numeric(),
                              stringsAsFactors = FALSE)

for (model in ml_models) {
  cat("\n\nProcessing model:", model, "\n")
  
  valid_cases <- !is.na(combined_data$human_majority) & !is.na(combined_data[[model]])
  
  human_labels <- combined_data$human_majority[valid_cases]
  model_labels <- combined_data[[model]][valid_cases]
  
  print_label_summary(human_labels, "Human majority")
  print_label_summary(model_labels, paste(model, "predictions"))
  
  # Calculate metrics
  cat("\nCalculating metrics for", model, ":\n")
  metrics <- tryCatch({
    calculate_metrics(human_labels, model_labels, combined_data[valid_cases, c("rater1", "rater2", "rater3", "rater4", "rater5")])
  }, warning = function(w) {
    cat("\nWarning in metric calculation:", conditionMessage(w))
    return(NULL)
  })
  
  if (!is.null(metrics)) {
    # Prepare row for metrics_results
    row <- data.frame(
      model = model,
      precision_positive = metrics$precision["Positive Emotion"],
      precision_negative = metrics$precision["Negative Emotion"],
      precision_neutral = metrics$precision["Neutral Expression"],
      precision_noface = metrics$precision["No face"],
      recall_positive = metrics$recall["Positive Emotion"],
      recall_negative = metrics$recall["Negative Emotion"],
      recall_neutral = metrics$recall["Neutral Expression"],
      recall_noface = metrics$recall["No face"],
      macro_f1 = metrics$macro_f1,
      weighted_f1 = metrics$weighted_f1,
      accuracy = metrics$accuracy,
      mcc = metrics$mcc,
      cohens_kappa = metrics$cohens_kappa,
      fleiss_kappa = metrics$fleiss_kappa,
      stringsAsFactors = FALSE
    )
    
    # Add the metrics to the results dataframe
    metrics_results <- rbind(metrics_results, row)
    
    # Print the metrics for the current model
    cat("\nMetrics for", model, ":\n")
    print(metrics)
  } else {
    cat("\nFailed to calculate metrics for", model, "\n")
  }
}

# Print and save results
print(metrics_results)
# write.csv(metrics_results, "results/ml_models_comparison.csv", row.names = FALSE)

# # Visualize results
# # Create a list to store the four plots
# metric_plots <- list()

# # Create a plot for each metric
# for (metric in c("macro_f1", "cohens_kappa", "fleiss_kappa")) {
#   metric_plots[[metric]] <- metrics_results %>%
#     ggplot(aes(x = !!sym(metric), y = model)) +
#     geom_bar(stat = "identity", fill = "steelblue") +
#     theme_minimal() +
#     theme(axis.text.y = element_text(hjust = 1)) +
#     labs(title = paste("Comparison of", gsub("_", " ", toupper(metric))),
#          x = "Score", y = "Model")
# }

# # Create separate plots for precision and recall
# precision_plot <- metrics_results %>%
#   select(model, starts_with("precision_")) %>%
#   pivot_longer(cols = -model, names_to = "category", values_to = "precision") %>%
#   mutate(category = gsub("precision_", "", category)) %>%
#   ggplot(aes(x = precision, y = model, fill = category)) +
#   geom_bar(stat = "identity", position = "dodge") +
#   theme_minimal() +
#   theme(axis.text.y = element_text(hjust = 1)) +
#   labs(title = "Comparison of Precision",
#        x = "Precision", y = "Model")

# recall_plot <- metrics_results %>%
#   select(model, starts_with("recall_")) %>%
#   pivot_longer(cols = -model, names_to = "category", values_to = "recall") %>%
#   mutate(category = gsub("recall_", "", category)) %>%
#   ggplot(aes(x = recall, y = model, fill = category)) +
#   geom_bar(stat = "identity", position = "dodge") +
#   theme_minimal() +
#   theme(axis.text.y = element_text(hjust = 1)) +
#   labs(title = "Comparison of Recall",
#        x = "Recall", y = "Model")

# # Save each plot separately
# ggsave("results/macro_f1_comparison.png", metric_plots[["macro_f1"]], width = 10, height = 6)
# ggsave("results/cohens_kappa_comparison.png", metric_plots[["cohens_kappa"]], width = 10, height = 6)
# ggsave("results/fleiss_kappa_comparison.png", metric_plots[["fleiss_kappa"]], width = 10, height = 6)
# ggsave("results/precision_comparison.png", precision_plot, width = 12, height = 6)
# ggsave("results/recall_comparison.png", recall_plot, width = 12, height = 6)

# # Print a message to confirm plots have been saved
# cat("\nPlots have been saved in the 'results' directory.\n")

# # Analyze label distribution
# label_distribution <- combined_data %>%
#   pivot_longer(cols = c(
#     "rater1",
#     "rater2",
#     "rater3",
#     "rater4",
#     "rater5",
#     ml_models
#   ), names_to = "rater", values_to = "label") %>%
#   group_by(rater, label) %>%
#   summarise(count = n()) %>%
#   group_by(rater) %>%
#   mutate(percentage = count / sum(count) * 100)

# label_distribution_plot <-
#   ggplot(label_distribution, aes(x = rater, y = percentage, fill = label)) +
#   geom_bar(stat = "identity", position = "stack") +
#   theme_minimal() +
#   labs(title = "Label Distribution by Rater", x = "Rater", y = "Percentage") +
#   scale_fill_brewer(palette = "Set3")
# save_plot(label_distribution_plot, "label_distribution")

# # Analyze agreement on "No face" cases
# no_face_agreement <- combined_data %>%
#   filter(
#     rater1 == "No face" | rater2 == "No face" | rater3 == "No face" |
#       rater4 == "No face" | rater5 == "No face" |
#       any(sapply(ml_models, function(model) combined_data[[model]] == "No face"))
#   ) %>%
#   mutate(agreement = rowSums(select(., rater1:rater5, ml_models) == "No face", na.rm = TRUE)) %>%
#   group_by(agreement) %>%
#   summarise(count = n()) %>%
#   mutate(percentage = count / sum(count) * 100)

# cat("\nAgreement on 'No face' cases:\n")
# print(no_face_agreement)

# no_face_agreement_plot <-
#   ggplot(no_face_agreement, aes(x = factor(agreement), y = percentage)) +
