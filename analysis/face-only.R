# Load required libraries
library(tidyverse)
library(irr)
library(ggplot2)
library(dplyr)

# Function to preprocess human coder data
preprocess_human_data <- function(data) {
  data %>%
    mutate(label = ifelse(face == "No", "No face", label))
}

# Function to preprocess ML data
preprocess_ml_data <- function(data) {
  if ("outcome" %in% names(data)) {
    # For models that use the 'outcome' column
    data %>%
      mutate(label = case_when(
        outcome == "PS" ~ "Positive Emotion",
        outcome == "NG" ~ "Negative Emotion",
        outcome == "NT" ~ "Neutral Expression",
        outcome == "NO" ~ "No face",
        TRUE ~ NA_character_
      )) %>%
      select(image_name, label)
  } else if ("label" %in% names(data)) {
    # For models that already have a 'label' column
    data %>%
      mutate(label = case_when(
        label == "Positive Emotion" ~ "Positive Emotion",
        label == "Negative Emotion" ~ "Negative Emotion",
        label == "Neutral Expression" ~ "Neutral Expression",
        label == "No face" ~ "No face",
        TRUE ~ NA_character_
      )) %>%
      select(image_name, label)
  } else {
    stop("Unexpected data format: neither 'outcome' nor 'label' column found")
  }
}

# Function to calculate F1 score
calculate_f1_score <- function(confusion_matrix) {
  precision <- diag(confusion_matrix) / colSums(confusion_matrix)
  recall <- diag(confusion_matrix) / rowSums(confusion_matrix)
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(mean(f1, na.rm = TRUE))
}

# Function to save ggplot
save_plot <- function(plot, filename) {
  png(paste0("figures/face/", filename, ".png"), width = 800, height = 600)
  print(plot)
  dev.off()
}

# Read and preprocess the human coder CSV files
data1 <- read.csv("results/C1.csv") %>% preprocess_human_data()
data2 <- read.csv("results/C2.csv") %>% preprocess_human_data()
data3 <- read.csv("results/C3.csv") %>% preprocess_human_data()
data4 <- read.csv("results/C4.csv") %>% preprocess_human_data()
data5 <- read.csv("results/C5.csv") %>% preprocess_human_data()

# Read and preprocess the ML algorithm CSV files
ml_models <- c("CLIP-1", "CLIP-2", "CLIP-1-emotion", "CLIP-emotion", "CNN-fer", 
               "CNN-strong-org", "CNN-strong", "CNN-weak", "CNN-weak-org")

ml_data <- list()
for (model in ml_models) {
  tryCatch({
    ml_data[[model]] <- read.csv(paste0("results/", model, ".csv")) %>% preprocess_ml_data()
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

# Filter cases where all raters and ML models agree a face is present
face_agreement_data <- combined_data %>%
  filter(
    rater1 != "No face" &
    rater2 != "No face" &
    rater3 != "No face" &
    rater4 != "No face" &
    rater5 != "No face" &
    if_all(all_of(ml_models), ~ . != "No face")
  )

# Update the consensus methods calculation
face_agreement_data <- face_agreement_data %>%
  rowwise() %>%
  mutate(
    human_consensus_all = if (length(unique(c(rater1, rater2, rater3, rater4, rater5))) == 1) {
      rater1
    } else {
      NA_character_
    },
    human_consensus_majority = {
      tab <- table(c(rater1, rater2, rater3, rater4, rater5))
      if (max(tab) >= 3) names(which.max(tab)) else NA_character_
    },
    human_consensus_any2 = {
      tab <- table(c(rater1, rater2, rater3, rater4, rater5))
      if (max(tab) >= 2) names(which.max(tab)) else NA_character_
    }
  ) %>%
  ungroup()

# Function to calculate metrics
calculate_metrics <- function(human_consensus, ml_labels) {
  valid_cases <- !is.na(human_consensus) & !is.na(ml_labels)
  human_consensus <- human_consensus[valid_cases]
  ml_labels <- ml_labels[valid_cases]

  agreement <- mean(human_consensus == ml_labels, na.rm = TRUE)
  kappa <- tryCatch(
    {
      kappa2(cbind(human_consensus, ml_labels))$value
    },
    error = function(e) {
      warning("Error in kappa calculation: ", e$message)
      NA
    }
  )

  confusion_matrix <- table(human_consensus, ml_labels)
  f1_score <- calculate_f1_score(confusion_matrix)

  list(
    agreement = agreement, kappa = kappa, f1_score = f1_score,
    confusion_matrix = confusion_matrix, valid_cases = sum(valid_cases)
  )
}

# Compare different human consensus methods with each ML model
consensus_methods <- c(
  "human_consensus_all",
  "human_consensus_majority",
  "human_consensus_any2"
)

consensus_comparison <- data.frame(
  method = character(),
  model = character(),
  agreement = numeric(),
  kappa = numeric(),
  f1_score = numeric(),
  valid_cases = numeric(),
  stringsAsFactors = FALSE
)

for (method in consensus_methods) {
  for (model in ml_models) {
    metrics <- calculate_metrics(
      face_agreement_data[[method]],
      face_agreement_data[[model]]
    )

    consensus_comparison <- rbind(
      consensus_comparison,
      data.frame(
        method = method,
        model = model,
        agreement = metrics$agreement,
        kappa = metrics$kappa,
        f1_score = metrics$f1_score,
        valid_cases = metrics$valid_cases
      )
    )

    cat("\nMetrics for", method, "vs", model, "(Face Agreement Cases):\n")
    print(metrics$confusion_matrix)
    cat("Agreement:", metrics$agreement, "\n")
    cat("Kappa:", metrics$kappa, "\n")
    cat("F1 Score:", metrics$f1_score, "\n")
    cat("Valid cases:", metrics$valid_cases, "\n")

    # Calculate and print precision and recall for each class
    precision <- diag(metrics$confusion_matrix) / colSums(metrics$confusion_matrix)
    recall <- diag(metrics$confusion_matrix) / rowSums(metrics$confusion_matrix)

    cat("Precision by class:\n")
    print(precision)

    cat("Recall by class:\n")
    print(recall)
  }
}

cat("\nComparison of Human Consensus Methods with ML Models (Face Agreement Cases):\n")
print(consensus_comparison)

# Visualize comparison of human consensus methods with ML models
consensus_comparison_plot <- ggplot(consensus_comparison, aes(x = method, y = agreement, fill = model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  theme_minimal() +
  labs(
    title = "Comparison of Human Consensus Methods with ML Models (Face Agreement Cases)",
    x = "Consensus Method", y = "Agreement"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(consensus_comparison_plot, "emotion_consensus_comparison")

# Analyze emotion label distribution
emotion_distribution <- face_agreement_data %>%
  pivot_longer(
    cols = c("rater1", "rater2", "rater3", "rater4", "rater5", all_of(ml_models)),
    names_to = "rater", values_to = "emotion"
  ) %>%
  group_by(rater, emotion) %>%
  summarise(count = n()) %>%
  group_by(rater) %>%
  mutate(percentage = count / sum(count) * 100)

emotion_distribution_plot <-
  ggplot(emotion_distribution, aes(x = rater, y = percentage, fill = emotion)) +
  geom_bar(stat = "identity", position = "stack") +
  theme_minimal() +
  labs(
    title = "Emotion Label Distribution by Rater and ML Model (Face Agreement Cases)",
    x = "Rater/Model", y = "Percentage"
  ) +
  scale_fill_brewer(palette = "Set3") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(emotion_distribution_plot, "emotion_label_distribution")

# Calculate pairwise Cohen's Kappa for emotion labels
calculate_cohens_kappa <- function(rater1, rater2) {
  kappa2(cbind(rater1, rater2))$value
}

emotion_kappa_results <- data.frame(
  comparison = character(),
  kappa = numeric(),
  stringsAsFactors = FALSE
)

raters <- c("rater1", "rater2", "rater3", "rater4", "rater5", ml_models)
for (i in 1:(length(raters) - 1)) {
  for (j in (i + 1):length(raters)) {
    kappa <- calculate_cohens_kappa(
      face_agreement_data[[raters[i]]], face_agreement_data[[raters[j]]]
    )
    emotion_kappa_results <- rbind(
      emotion_kappa_results, data.frame(
        comparison = paste(raters[i], "vs", raters[j]), kappa = kappa
      )
    )
  }
}

cat("\nPairwise Cohen's Kappa for Emotion Labels (Face Agreement Cases):\n")
print(emotion_kappa_results)

# Visualize pairwise Cohen's Kappa for emotion labels
emotion_kappa_plot <-
  ggplot(emotion_kappa_results, aes(x = comparison, y = kappa)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  labs(
    title = "Pairwise Cohen's Kappa for Emotion Labels (Face Agreement Cases)",
    x = "Rater/Model Comparison", y = "Kappa Value"
  ) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
save_plot(emotion_kappa_plot, "emotion_pairwise_kappa")

# Calculate Fleiss' Kappa for emotion labels
emotion_fleiss_kappa <- kappam.fleiss(face_agreement_data[, c("rater1", "rater2", "rater3", "rater4", "rater5", ml_models)])
cat("\nFleiss' Kappa for Emotion Labels (Face Agreement Cases):", emotion_fleiss_kappa$value, "\n")

cat("\nSample data for verification (Face Agreement Cases):\n")
print(face_agreement_data %>%
  select(rater1, rater2, rater3, rater4, rater5, all_of(ml_models),
         human_consensus_all, human_consensus_majority, human_consensus_any2) %>%
  head(20))

# Print summary of consensus methods
cat("\nSummary of consensus methods (Face Agreement Cases):\n")
print(face_agreement_data %>%
  summarise(
    total_cases = n(),
    all_consensus = sum(!is.na(human_consensus_all)),
    majority_consensus = sum(!is.na(human_consensus_majority)),
    any2_consensus = sum(!is.na(human_consensus_any2))
  ))