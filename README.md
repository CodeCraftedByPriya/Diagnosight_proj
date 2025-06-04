## *Healthcare Data Analysis Project*
This project focuses on cleaning, transforming, and analyzing a healthcare dataset to extract insights about patient characteristics, treatment patterns, and recovery outcomes. It includes data preprocessing, feature engineering, scaling, and visual analysis using Python libraries like Pandas, NumPy, Seaborn, and Matplotlib.

## *Data Preprocessing*
- **Missing Values:**  
  - Numerical columns filled with median  
  - Categorical columns filled with mode  

## *Data Preprocessing*
- Loaded healthcare.csv into a Pandas dataframe.
- Handled missing values by imputing medians for numerical columns and modes for categorical columns.
- Detected and capped outliers using the Interquartile Range (IQR) method.
- **Engineered features:**
    - Converted Recovery_Time and Treatment_Duration into numeric columns (Recovery_Days, Treatment_Days)
    - Split Blood_Pressure into Systolic and Diastolic.
    - Created an Age_Group categorical variable (Child, Young Adult, Adult, Senior).
- Applied StandardScaler to numeric features such as Age, Heart Rate, Temperature, SAT score, and recovery/treatment days.

## *Exploratory Data Analysis (EDA)*
- Visualized distributions for Age, Treatment Days, Recovery Days, and Satisfaction (SAT) Scores.
- **Explored group-level patterns:**
  - SAT by Age Group
  - Treatment and Recovery Days by Surgery Type
  - Recovery Days by Gender
  - Average Recovery Days by Diagnosis
- Created count plots for Gender and Surgery Type, bar plots for top diagnoses, and correlation heatmaps of numeric variables.

## *Diagnostic Analysis*
- Analyzed impact of family history and allergies on diagnosis outcomes.
- Investigated associations between surgery types and treatment/recovery durations.
- Explored hospital-wise differences in recovery times for common diagnoses.
- Used visualizations and statistical summaries to support findings.

## *Predictive Analysis*
**Scenario 1: Diagnosis Prediction**
- Encoded categorical targets (Diagnosis) and features (Gender).
- Trained a Random Forest Classifier on patient metrics (Age, Gender, Heart Rate, Temperature).
- Evaluated the model with classification reports.
- Predicted diagnosis for a sample patient (Mr. Harry, 70 years old, male) based on input health parameters.

**Scenario 2: Recovery Time Prediction**
- Built a regression model (Random Forest Regressor) to predict recovery days.
- Used patient demographics and diagnosis (encoded) as inputs.
- Predicted recovery duration for a sample patient (Ms. Reena, 40 years old, female, diagnosed with influenza).

## *Sentiment Analysis on Patient Feedback*
- Performed NLP on the Feedback column using TextBlob.
- Calculated polarity scores to measure sentiment.
- **Categorized sentiments into:**
  - Positive (polarity > 0)
  - Neutral (polarity = 0)
  - Negative (polarity < 0)
- Visualized sentiment distribution using a pie chart to represent patient satisfaction levels.
