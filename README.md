## *Healthcare Data Analysis Project*
This project focuses on cleaning, transforming, and analyzing a healthcare dataset to extract insights about patient characteristics, treatment patterns, and recovery outcomes. It includes data preprocessing, feature engineering, scaling, and visual analysis using Python libraries like Pandas, NumPy, Seaborn, and Matplotlib.

## *Data Preprocessing*
- **Missing Values:**  
  - Numerical columns filled with median  
  - Categorical columns filled with mode  

- **Outliers:**  
  - Detected and capped using the IQR method  

- **Feature Engineering:**  
  - `Recovery_Time` and `Treatment_Duration` converted into numeric `Recovery_Days` and `Treatment_Days`  
  - `Blood_Pressure` split into `Systolic` and `Diastolic` values  
  - Age groups created as `Child`, `Young Adult`, `Adult`, `Senior`  

- **Scaling:**  
  - Standard scaling applied to numeric features (`Age`, `Heart_Rate`, `Temperature`, `SAT`, `Treatment_Days`, `Recovery_Days`, `Systolic`, `Diastolic`) using `StandardScaler`

## Exploratory Data Analysis (EDA)

The analysis includes:

### Distributions
- Age, Treatment Days, Recovery Days, SAT Score

### Group-Level Trends
- SAT by Age Group  
- Treatment and Recovery Days by Surgery Type  
- Recovery Days by Gender  
- Average Recovery by Diagnosis  

### Visualizations
- Count plots for Gender and Surgery Type  
- Bar plots for Top Diagnoses  
- Box and violin plots for SAT and Recovery patterns  
- Correlation heatmap for numeric variables  
- Scatter plot showing relation between Treatment and Recovery days by Age Group
