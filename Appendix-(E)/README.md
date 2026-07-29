- **Project Name**：  Feature Weighted Fusion using SLSQP

- **Project Purpose**： This project aims to perform feature weighted fusion using the SLSQP optimization algorithm. By assigning appropriate weights to each feature in a dataset with multiple features, we can extract deeper-level features and reduce the computational complexity of models.

- **Main Features**：

  - Utilize the SLSQP algorithm for feature weighted fusion
  - Extract more representative and deeper-level features
  - Reduce computational complexity of models while maintaining performance
  
- **Environment Dependencies**：

  **System Environment**：

  - Operating System: Windows, macOS, or Linux
  - Python Version: 3.7 or above

  **Software Dependencies**：

  - numpy: For numerical computations
  - pandas: For data manipulation and analysis
  - scipy: For optimization algorithms
  - scikit-learn: For data preprocessing
  
- **Installation and Running Guide**：

  - Installing Dependencies

    Install Required Python Packages:

    ```Python
    pip install numpy pandas scipy scikit-learn
    ```

  - Running the Project

    ```python
    python feature_fusion.py
    ```

    Replace `feature_fusion.py` with the name of your Python script file. (Simple is "Calculation_SLSQP.py")

  - Prepare the Data File

    Make sure your data file (in CSV format) is ready and named `Data.csv` (or modify the filename in the code to match your filename).

  - View the Results

    After running the script, two CSV files will be generated in the project directory:

    1. Optimized_Weights.csv:
       - Contains the optimized weights for each feature.
       - Each row represents a feature and its corresponding weight.
    2. Fused_Features.csv:
       - Contains the fused feature values for each sample.
       - Each row represents a sample and its fused feature value.

- **Directory Structure Description**：

  ```python
  project_directory/
  │
  ├── FeatureData.csv        # Input feature data file
  ├── feature_fusion.py      # Main Python script
  ├── Optimized_Weights.csv  # Output file containing optimized weights
  ├── Fused_Features.csv     # Output file containing fused feature values
  └── README.md              # This documentation file
  ```

- **Version Update Summary**：

  | Version | Update Date | Update Content                                               |
  | :-----: | :---------: | ------------------------------------------------------------ |
  |  1.0.0  | 2026-06-28  | Initial version, implementing feature weighted fusion using the SLSQP algorithm |

- **Usage Instructions**：

  1. Function Description

     This project uses the SLSQP optimization algorithm to assign weights to features in a dataset. The goal is to find the optimal weight combination that maximizes the variance of the weighted features. This helps in extracting more representative and deeper-level features while reducing the computational complexity of models.

  2. Parameter Adjustment

     - **Data File**: Modify the filename in `data = pd.read_csv('FeatureData.csv', encoding='UTF-8-sig')` to load a different dataset.
     - **Optimization Method**: You can try different optimization methods by changing the `method` parameter in the `minimize` function.
     - **Initial Weights**: Adjust the initial weights by modifying the `initial_weights` variable.

  3. Troubleshooting

     - **Data Format Issues**: Ensure the input data is in CSV format with each column representing a feature.
     - **Dependency Issues**: If you encounter missing package issues, use `pip install` to install the required packages.
     - **Optimization Failures**: If optimization fails, try different initial weights or optimization methods.

