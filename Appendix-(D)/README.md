- **Project Name**： LightGBM Information Gain Calculator

- **Project Purpose**： This project aims to calculate the information gain of each feature in a dataset for a multi-class classification task using the LightGBM framework and visualize the importance of each feature in different classes in the form of a bar chart.

- **Main Features**：

  - Train a multi-class model using LightGBM
  - Calculate feature importance (information gain)
  - Convert information gain to percentage form
  - Visualize the information gain percentage of each feature in different classes

- **Environment Dependencies**：

  **System Environment**：

  - Operating System: Windows, macOS, or Linux
  - Python Version: 3.7 or above

  **Software Dependencies**：

  - pandas: For data processing and analysis
  - lightgbm: For training multi-class models
  - matplotlib: For visualizing information gain
  - numpy: For numerical computations
  - scikit-learn: For dataset splitting

- **Installation and Running Guide**：

  - Installing Dependencies

    Create a Virtual Environment (optional but recommended):

    ```Python
    python -m venv lightgbm_env
    ```

    Activate the virtual environment:

    ```python
    # Windows
    lightgbm_env\Scripts\activate
    
    # macOS/Linux
    source lightgbm_env/bin/activate
    ```

  - Install the Required Python Packages

    ```Python
    pip install pandas lightgbm matplotlib numpy scikit-learn
    ```

  - Prepare the Data File

    1. Make sure your data file (in CSV format) is ready and named `Data.csv` (or modify the filename in the code to match your filename).
    2. The last column of the data file should be the classification column, with the preceding columns being feature columns.

  - Run the Script

    ```python
    python your_script_name.py
    ```

    Replace "your_script_name.py" with the name of your Python script file. (Simple is "Calculate_InformationGain_using_LightGBM.py")

  - View the Results

    After running the script, the information gain chart will be saved as the file "information_gain.png".

- **Directory Structure Description**：

  ```python
  project_directory/
  │
  ├── Data.csv               # Input data file
  ├── your_script_name.py    # Main Python script
  └── README.md              # This documentation file
  ```

- **Version Update Summary**：

  | Version | Update Date | Update Content                                               |
  | :-----: | :---------: | ------------------------------------------------------------ |
  |  1.0.0  | 2026-06-08  | Initial version, implementing the functionality of training a LightGBM multi-class model and calculating and visualizing feature information gain |

- **Usage Instructions**：

  1. Function Description

     This tool trains a multi-class model using the LightGBM framework and calculates the information gain of each feature. Information gain reflects the importance of a feature to the classification task. By visualizing the information gain percentage of each feature in different classes, users can intuitively understand which features are more important to which classes.

  2. Parameter Adjustment

     - **Data File**: Modify the filename in `data = pd.read_csv('Data.csv', encoding='UTF-8-sig')` to load a different dataset.
     - **LightGBM Parameters**: Adjust the LightGBM training parameters in the `params` dictionary to meet the needs of different datasets and tasks.
     - **Chart Save Path**: Modify the filename in `plt.savefig('information_gain.png')` to change the save path and name of the output chart.

  3. Troubleshooting

     - **Data Format Issues**: Ensure the input data is in CSV format with the last column being the classification column.
     - **Dependency Conflicts**: If you encounter dependency conflicts, try installing the project dependencies in a new virtual environment.
     - **Chart Display Issues**: If the chart is not displayed correctly, try adjusting the Matplotlib backend settings or updating Matplotlib to the latest version.

