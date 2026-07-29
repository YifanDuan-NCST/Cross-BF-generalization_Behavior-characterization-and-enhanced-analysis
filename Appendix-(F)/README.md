- **Project Name**：  PCA Feature Reduction and Extraction

- **Project Purpose**： This project aims to perform dimensionality reduction and deep feature extraction on the `BetterFeatureSet.csv` dataset using the PCA (Principal Component Analysis) algorithm. By reducing the number of features while retaining the most important information, we can improve the efficiency and performance of machine learning models.

- **Main Features**：

  - Utilize PCA for dimensionality reduction
  - Extract deep-level features from the dataset
  - Retain over 85% of the cumulative explained variance
  - Save the reduced feature set and visualization results
  
- **Environment Dependencies**：

  **System Environment**：

  - Operating System: Windows, macOS, or Linux
  - Python Version: 3.7 or above

  **Software Dependencies**：

  - numpy: For numerical computations
  - pandas: For data manipulation and analysis
  - scikit-learn: For PCA implementation
  - matplotlib: For data visualization
  
- **Installation and Running Guide**：

  - Installing Dependencies

    Install Required Python Packages:

    ```Python
    pip install numpy pandas scikit-learn matplotlib
    ```

  - Prepare the Data File

    - Make sure your data file (in CSV format) is ready and named `Data.csv` (or modify the filename in the code to match your filename).
    - Each column should represent a feature, and each row should represent a sample.
    
  - Running the Project

    ```python
    python pca_feature_reduction.py
    ```

    Replace `pca_feature_reduction.py` with the name of your Python script file. (Simple is "Calculation_PCA.py")

  - View the Results

    After running the script, two files will be generated in the project directory:

    1. BestFeatureSet.csv:
       - Contains the reduced feature set after PCA transformation.
       - Each column represents a principal component (PC1, PC2, ...).
    2. pca_cumulative_variance.png:
       - A plot showing the cumulative explained variance ratio as a function of the number of principal components.
       - Helps visualize the number of components needed to achieve the desired cumulative explained variance ratio (over 85%).

- **Directory Structure Description**：

  ```python
  project_directory/
  │
  ├── BetterFeatureSet.csv    # Input feature data file
  ├── pca_feature_reduction.py # Main Python script
  ├── BestFeatureSet.csv      # Output file containing the reduced feature set
  ├── pca_cumulative_variance.png # Output plot showing cumulative explained variance
  └── README.md               # This documentation file
  ```

- **Version Update Summary**：

  | Version | Update Date | Update Content                                               |
  | :-----: | :---------: | ------------------------------------------------------------ |
  |  1.0.0  | 2026-06-08  | Initial version, implementing PCA-based feature reduction and extraction |

- **Usage Instructions**：

  1. Function Description

     This project uses the PCA algorithm to reduce the dimensionality of the feature dataset while retaining the most important information. The goal is to extract deep-level features that can improve the efficiency and performance of machine learning models.

  2. Parameter Adjustment

     - **Data File**: Modify the filename in `data = pd.read_csv('BetterFeatureSet.csv', encoding='UTF-8-sig')` to load a different dataset.
     - **Cumulative Explained Variance Ratio**: Adjust the threshold (currently 85%) by modifying the value in `np.argmax(cumulative_explained_variance_ratio > 0.85) + 1`.
     
3. Troubleshooting
  
   - **Data Format Issues**: Ensure the input data is in CSV format with each column representing a feature.
     - **Dependency Issues**: If you encounter missing package issues, use `pip install` to install the required packages.
     - **Visualization Issues**: If the plot is not generated, ensure you have the necessary permissions to write files to the project directory.
