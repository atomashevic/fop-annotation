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

preprocess_ml_data <- function(data) {
  data %>%
    mutate(label = case_when(
      outcome == "PS" ~ "Positive Emotion",
      outcome == "NG" ~ "Negative Emotion",
      outcome == "NT" ~ "Neutral Expression",
      outcome == "NO" ~ "No face",
      TRUE ~ NA_character_
    )) %>%
    select(image_name, label)
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

# Read and preprocess the human coder CSV files
data1 <- read.csv("results/C1.csv") %>% preprocess_human_data()
data2 <- read.csv("results/C2.csv") %>% preprocess_human_data()
data3 <- read.csv("results/C3.csv") %>% preprocess_human_data()
data4 <- read.csv("results/C4.csv") %>% preprocess_human_data()
data5 <- read.csv("results/C5.csv") %>% preprocess_human_data()

# Read and preprocess the ML algorithm CSV file
ml <- read.csv("results/ML.csv") %>% preprocess_ml_data()

print(ml) # Combine the data into a single dataframe

combined_data <- data1 %>%
  select(image_name, label) %>%
  rename(rater1 = label) %>%
  left_join(data2 %>% select(image_name, label) %>% rename(rater2 = label), by = "image_name") %>%
  left_join(data3 %>% select(image_name, label) %>% rename(rater3 = label), by = "image_name") %>%
  left_join(data4 %>% select(image_name, label) %>% rename(rater4 = label), by = "image_name") %>%
  left_join(data5 %>% select(image_name, label) %>% rename(rater5 = label), by = "image_name") %>%
  left_join(ml %>% rename(ml = label), by = "image_name")

# Update the consensus methods calculation
combined_data <- combined_data %>%
  rowwise() %>%
  mutate(
    human_consensus_all = if (length(unique(c(rater1, rater2, rater3, rater4, rater5))) == 1) rater1 else NA_character_,
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

# Calculate pairwise Cohen's Kappa
calculate_cohens_kappa <- function(rater1, rater2) {
  kappa2(cbind(rater1, rater2))$value
}

kappa_results <- data.frame(
  comparison = character(),
  kappa = numeric(),
  stringsAsFactors = FALSE
)

raters <- c("rater1", "rater2", "rater3", "rater4", "rater5", "ml")
for (i in 1:(length(raters) - 1)) {
  for (j in (i + 1):length(raters)) {
    kappa <- calculate_cohens_kappa(combined_data[[raters[i]]], combined_data[[raters[j]]])
    kappa_results <- rbind(kappa_results, data.frame(comparison = paste(raters[i], "vs", raters[j]), kappa = kappa))
  }
}

# Calculate Fleiss' Kappa
fleiss_kappa <- kappam.fleiss(combined_data[, c("rater1", "rater2", "rater3", "rater4", "rater5", "ml")])

fleiss_kappa
# Calculate percent agreement
calculate_percent_agreement <- function(rater1, rater2) {
  sum(rater1 == rater2, na.rm = TRUE) / sum(!is.na(rater1) & !is.na(rater2))
}

percent_agreement_results <- data.frame(
  comparison = character(),
  agreement = numeric(),
  stringsAsFactors = FALSE
)

for (i in 1:(length(raters) - 1)) {
  for (j in (i + 1):length(raters)) {
    agreement <- calculate_percent_agreement(
      combined_data[[raters[i]]],
      combined_data[[raters[j]]]
    )
    percent_agreement_results <- rbind(
      percent_agreement_results,
      data.frame(
        comparison = paste(
          raters[i],
          "vs",
          raters[j]
        ),
        agreement = agreement
      )
    )
  }
}

# Calculate overall percent agreement
overall_agreement <- mean(sapply(1:nrow(combined_data), function(i) {
  length(unique(unlist(combined_data[i, c(
    "rater1",
    "rater2",
    "rater3",
    "rater4",
    "rater5",
    "ml"
  )]))) == 1
}), na.rm = TRUE)

# Visualize pairwise Cohen's Kappa
kappa_plot <- ggplot(kappa_results, aes(x = comparison, y = kappa)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  theme_minimal() +
  labs(
    title = "Pairwise Cohen's Kappa",
    x = "Rater Comparison",
    y = "Kappa Value"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

kappa_plot
save_plot(kappa_plot, "pairwise_cohens_kappa")

# Visualize percent agreement
percent_agreement_plot <-
  ggplot(
    percent_agreement_results,
    aes(x = comparison, y = agreement)
  ) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  theme_minimal() +
  labs(
    title = "Pairwise Percent Agreement",
    x = "Rater Comparison",
    y = "Percent Agreement"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_plot(percent_agreement_plot, "pairwise_percent_agreement")

# Print results
cat("Pairwise Cohen's Kappa:\n")
print(kappa_results)

cat("\nFleiss' Kappa:", fleiss_kappa$value, "\n")

cat("\nPairwise Percent Agreement:\n")
print(percent_agreement_results)

cat("\nOverall Percent Agreement:", overall_agreement, "\n")

# Analyze label distribution
label_distribution <- combined_data %>%
  pivot_longer(cols = c(
    "rater1",
    "rater2",
    "rater3",
    "rater4",
    "rater5",
    "ml"
  ), names_to = "rater", values_to = "label") %>%
  group_by(rater, label) %>%
  summarise(count = n()) %>%
  group_by(rater) %>%
  mutate(percentage = count / sum(count) * 100)

label_distribution_plot <-
  ggplot(label_distribution, aes(x = rater, y = percentage, fill = label)) +
  geom_bar(stat = "identity", position = "stack") +
  theme_minimal() +
  labs(title = "Label Distribution by Rater", x = "Rater", y = "Percentage") +
  scale_fill_brewer(palette = "Set3")
save_plot(label_distribution_plot, "label_distribution")

# Analyze agreement on "No face" cases
no_face_agreement <- combined_data %>%
  filter(
    rater1 == "No face" | rater2 == "No face" | rater3 == "No face" |
      rater4 == "No face" | rater5 == "No face" | ml == "No face"
  ) %>%
  mutate(agreement = rowSums(select(., rater1:ml) == "No face", na.rm = TRUE)) %>%
  group_by(agreement) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

cat("\nAgreement on 'No face' cases:\n")
print(no_face_agreement)

no_face_agreement_plot <-
  ggplot(no_face_agreement, aes(x = factor(agreement), y = percentage)) +
  geom_bar(stat = "identity", fill = "orange") +
  theme_minimal() +
  labs(
    title = "Agreement on 'No face' Cases",
    x = "Number of Raters Agreeing", y = "Percentage"
  ) +
  scale_x_discrete(labels = function(x) paste(x, "Raters"))
save_plot(no_face_agreement_plot, "no_face_agreement")

# Compare different human consensus methods with ML
consensus_methods <- c(
  "human_consensus_all",
  "human_consensus_majority",
  "human_consensus_any2"
)

consensus_comparison <- data.frame(
  method = character(),
  agreement = numeric(),
  kappa = numeric(),
  f1_score = numeric(),
  valid_cases = numeric(),
  stringsAsFactors = FALSE
)

for (method in consensus_methods) {
  valid_cases <- !is.na(combined_data[[method]]) & !is.na(combined_data$ml)
  human_consensus <- combined_data[[method]][valid_cases]
  ml_labels <- combined_data$ml[valid_cases]

  agreement <- mean(human_consensus == ml_labels, na.rm = TRUE)
  kappa <- calculate_cohens_kappa(human_consensus, ml_labels)

  confusion_matrix <- table(human_consensus, ml_labels)
  f1_score <- calculate_f1_score(confusion_matrix)

  consensus_comparison <- rbind(
    consensus_comparison,
    data.frame(
      method = method,
      agreement = agreement,
      kappa = kappa,
      f1_score = f1_score,
      valid_cases = sum(valid_cases)
    )
  )

  cat("\nMetrics for", method, "vs ML:\n")
  print(confusion_matrix)
  cat("Agreement:", agreement, "\n")
  cat("Kappa:", kappa, "\n")
  cat("F1 Score:", f1_score, "\n")
  cat("Valid cases:", sum(valid_cases), "\n")

  # Calculate and print precision and recall for each class
  precision <- diag(confusion_matrix) / colSums(confusion_matrix)
  recall <- diag(confusion_matrix) / rowSums(confusion_matrix)

  cat("Precision by class:\n")
  print(precision)

  cat("Recall by class:\n")
  print(recall)
}

cat("\nComparison of Human Consensus Methods with ML:\n")
print(consensus_comparison)

consensus_comparison_plot <- ggplot(consensus_comparison, aes(x = method)) +
  geom_bar(
    aes(y = agreement, fill = "Agreement"),
    stat = "identity", position = position_dodge()
  ) +
  geom_bar(
    aes(y = kappa, fill = "Kappa"),
    stat = "identity", position = position_dodge()
  ) +
  geom_bar(
    aes(y = f1_score, fill = "F1 Score"),
    stat = "identity", position = position_dodge()
  ) +
  theme_minimal() +
  labs(
    title = "Comparison of Human Consensus Methods with ML",
    x = "Consensus Method", y = "Score"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c(
    "Agreement" = "purple",
    "Kappa" = "pink",
    "F1 Score" = "orange"
  ))
save_plot(consensus_comparison_plot, "consensus_comparison")

# Add this section to print out some sample data for verification
cat("\nSample data for verification:\n")
print(head(combined_data %>% select(
  rater1,
  rater2,
  rater3,
  rater4,
  rater5,
  ml,
  human_consensus_all,
  human_consensus_majority,
  human_consensus_any2
), 20))

# Print summary of consensus methods
cat("\nSummary of consensus methods:\n")
print(combined_data %>%
  summarise(
    all_consensus = sum(!is.na(human_consensus_all)),
    majority_consensus = sum(!is.na(human_consensus_majority)),
    any2_consensus = sum(!is.na(human_consensus_any2))
  ))
