- **Project Name**： Parameter-Class Correlation Analyzer

- **Project Purpose**： This project aims to analyze the correlation between parameters and class labels in a dataset. It calculates the Spearman's rank correlation between each parameter and each class label, then visualizes the results using a correlation matrix and bar plots.

- **Main Features**：

  - Calculate Spearman's rank correlation coefficient correlations between parameters and class labels
  - Visualize correlations using a matrix and bar plots
  - Provide intuitive insights into parameter importance for different classes
  
- **Environment Dependencies**：

  **System Environment**：

  - Operating System: Windows, macOS, or Linux
  - R Version: 3.6 or above

  **Software Dependencies**：

  - corrplot: For correlation matrix visualization
  - ggplot2: For data visualization
  - reshape2: For data reshaping
  
- **Installation and Running Guide**：

  - Installing Dependencies

    Install Required R Packages:

    ```R
    install.packages("corrplot")
    install.packages("ggplot2")
install.packages("reshape2")
    ```

  - Run the Script
  
    ```R
    source("your_script_name.R")
    ```
  
    Replace `your_script_name.R` with the name of your R script file. (Simple is "Calculation_of_spearman's_rank_correlation_coefficient.R")

- **Directory Structure Description**：

  ```R
  project_directory/
  │
  ├── DataSet.csv           # Input data file
  ├── your_script_name.R    # Main R script
  └── README.md             # This documentation file
  ```

- **Version Update Summary**：

  | Version | Update Date | Update Content                                               |
  | :-----: | :---------: | ------------------------------------------------------------ |
  |  1.0.0  | 2025-06-08  | Initial version, implementing the functionality of calculating and visualizing parameter-class correlations |

- **Usage Instructions**：

  1. Function Description

     This tool calculates the Spearman's rank correlation between each parameter and each class label in a dataset. It then visualizes the correlations using a matrix plot and a bar plot. The correlation matrix provides a comprehensive view of how each parameter relates to each class, while the bar plot highlights the correlations for each parameter across different classes.

  2. Parameter Adjustment

     - **Data File**: Modify the filename in `data <- read.csv('DataSet.csv', encoding = "UTF-8-sig")` to load a different dataset.
     - **Correlation Method**: The script uses the Spearman correlation method. You can adjust other parameters in the `cor` function if needed.
     - **Visualization Customization**: Adjust the visualization parameters in the `corrplot` and `ggplot` functions to customize the appearance of the plots.

  3. Troubleshooting

     - **Data Format Issues**: Ensure the input data is in CSV format with the last column being the classification column.
     - **Dependency Issues**: If you encounter issues with missing packages, use the `install.packages()` function to install the required packages.
     - **Plot Customization**: If the plots are not displayed as expected, modify the parameters in the visualization functions to suit your preferences.

