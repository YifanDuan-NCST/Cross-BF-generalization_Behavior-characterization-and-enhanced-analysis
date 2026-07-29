# Loading necessary packages
library(corrplot)  # For correlation matrix visualization
library(ggplot2)   # For data visualization
library(reshape2)  # For data reshaping

# Setting seed for reproducibility
set.seed(123)

# Reading the dataSet
data <- read.csv('DataSet.csv', encoding = "UTF-8-sig")
data$level <- factor(data$level)

# Extracting parameter columns and the classification column
params <- data[, -ncol(data)]  # Extract all columns except the last one as parameters
class_labels <- levels(data$level)

# Getting parameter names as labels
param_labels <- colnames(params)

# Initializing correlation matrix
num_classes <- length(class_labels)
num_params <- ncol(params)
correlation_matrix <- matrix(0, nrow = num_params, ncol = num_classes)
rownames(correlation_matrix) <- param_labels
colnames(correlation_matrix) <- class_labels

# Calculating correlations
for (param_idx in 1:num_params) {
  param_data <- params[, param_idx]
  
  for (class_idx in seq_along(class_labels)) {
    # Creating binary class data (1 for the current class, 0 otherwise)
    class_data <- as.numeric(data$level == class_labels[class_idx])
    
    # Calculating Pearson correlation
    correlation <- cor(param_data, class_data, method = "spearman")
    correlation_matrix[param_idx, class_idx] <- correlation
  }
}

# Transposing the correlation matrix to have classes as rows and parameters as columns
correlation_matrix_transposed <- t(correlation_matrix)

# Creating correlation matrix plot
corrplot(correlation_matrix_transposed, method = 'square', 
         col = colorRampPalette(c('blue', 'white', 'red'))(200), 
         title = 'Parameter-Class Correlation Matrix',
         tl.col = 'black', tl.srt = 45)

# Creating a bar plot for each parameter's correlation across classes
bar_data <- melt(correlation_matrix)
colnames(bar_data) <- c('Parameter', 'Class', 'Correlation')

# Bar plot
ggplot(bar_data, aes(x = reorder(Parameter, Correlation), y = Correlation, fill = Class)) +
  geom_bar(stat = 'identity', position = 'dodge') +
  coord_flip() +
  scale_fill_distiller(palette = 'RdBu') +
  labs(title = 'Parameter-Class Correlation Bar Plot', x = 'Parameter', y = 'Correlation') +
  theme_minimal()
