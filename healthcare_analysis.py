import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import chi2_contingency, pearsonr
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px


# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

# DATA CLEANING AND ENCODING
df.drop(columns=['Patient_ID', 'Name', 'Address'], errors='ignore', inplace=True)

# Convert Recovery_Time and Treatment_Duration to numeric (in days)
def convert_duration(duration):
    if pd.isnull(duration):
        return np.nan
    duration = duration.lower().strip()
    if 'day' in duration:
        return int(duration.split()[0])
    elif 'week' in duration:
        return int(duration.split()[0]) * 7
    elif 'month' in duration:
        return int(duration.split()[0]) * 30
    elif 'year' in duration:
        return int(duration.split()[0]) * 365
    return np.nan

# New columns with the converted Time
df['Recovery_Days'] = df['Recovery_Time'].apply(convert_duration)
df['Treatment_Days'] = df['Treatment_Duration'].apply(convert_duration)

# Extract systolic and diastolic from Blood_Pressure
df[['Systolic', 'Diastolic']] = df['Blood_Pressure'].str.split('/', expand=True).astype(float)

# Create Age_Group column
def age_group(age):
    if age < 18:
        return 'Child'
    elif age < 40:
        return 'Young Adult'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'

# New column named 'Age_Group'
df['Age_Group'] = df['Age'].apply(age_group)

# Convert 'FamilyHistory' to binary: 'Yes' -> 1, 'No' -> 0
df['FamilyHistory'] = df['FamilyHistory'].map({'Yes': 1, 'No': 0})

# Create a binary column for hypertension risk
df['Hypertension_Risk'] = np.where((df['Systolic'] >= 130) | (df['Diastolic'] >= 80), 1, 0)

# Drop the original 'Blood_Pressure' column
df.drop('Blood_Pressure', axis=1, inplace=True)

# Convert 'SAT' to numeric, coerce errors to NaN
df['SAT'] = pd.to_numeric(df['SAT'], errors='coerce')

# Fill missing values
# Numeric columns - fill missing with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns - fill missing with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


# Encode categorical variables
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['X-ray_Results'] = df['X-ray_Results'].map({'Normal': 0, 'Abnormal': 1})
df['Allergies'] = df['Allergies'].map({'No': 0, 'Yes': 1})

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'Diagnosis', 'Medication']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target variable
X = df.drop(columns=['Diagnosis', 'Recovery_Days'])
y_diagnosis = df['Diagnosis']
y_recovery = df['Recovery_Days']

# Fill missing values with median for numeric columns
df.fillna(df.median(numeric_only=True), inplace=True)


# Handling Outliers - Using IQR
def handle_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    median = df[column].median()
    df[column] = np.where((df[column] < lower) | (df[column] > upper), median, df[column])

for col in numeric_cols:
    handle_outliers(col)


# Define scale_cols
scale_cols = ['Age', 'Heart_Rate', 'Temperature', 'SAT', 'Treatment_Days', 'Recovery_Days', 'Systolic', 'Diastolic']

# Standardize the scale_cols
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[scale_cols]), columns=[col + '_scaled' for col in scale_cols])
df = pd.concat([df, df_scaled], axis=1)


# Set default Seaborn theme and palette
sns.set(style="whitegrid")
sns.set_palette("pastel")

# EDA
# Distribution of Age, Treatment Days, Recovery Days, and SAT
df[['Age', 'Treatment_Days', 'Recovery_Days', 'SAT']].hist(figsize=(10, 6), bins=20)
plt.suptitle("Distributions of Age, Treatment & Recovery Days, and SAT")
plt.tight_layout()
plt.show()

# Heart Rate vs Diagnosis
plt.figure(figsize=(6,4))
sns.boxplot(x='Diagnosis', y='Heart_Rate', data=df)
plt.title("Heart Rate across Diagnoses")
plt.show()
# Insight: Certain diagnoses show significant deviations in heart rate.

# Temperature vs Diagnosis
plt.figure(figsize=(6,4))
sns.violinplot(x='Diagnosis', y='Temperature', data=df)
plt.title("Temperature by Diagnosis Type")
plt.show()
# Insight: Some illnesses like infections may show higher temperatures.

# Correlation heatmap
num_cols = ['Age', 'Systolic', 'Diastolic', 'Heart_Rate', 'Temperature', 'Treatment_Days', 'Recovery_Days', 'SAT', 'FamilyHistory']
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm'); plt.title("Correlation Heatmap"); plt.show()

# Scatterplot: Treatment vs Recovery
sns.scatterplot(data=df, x='Treatment_Days', y='Recovery_Days', hue='Age_Group')
plt.title("Treatment Duration vs Recovery Time by Age Group")
plt.xlabel("Treatment Days")
plt.ylabel("Recovery Days")
plt.show()

# Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Gender', palette='Set2')
plt.title("Patient Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Diagnosis Distribution (Top 10)
top_diagnoses = df['Diagnosis'].value_counts().head(10)
print("\nTop 10 Diagnoses:")
print(top_diagnoses)

# Boxplot: SAT Score by Age Group
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='Age_Group', y='SAT', palette='Pastel1')
plt.title("SAT Scores by Age Group")
plt.ylabel("Satisfaction Score")
plt.show()

# Age distribution by Gender
plt.figure(figsize=(10, 6))
sns.violinplot(x='Gender', y='Age', data=df, inner='quartile', palette='pastel')
plt.title('Age Distribution by Gender')
plt.show()
df.head()

# Count of Surgery Types
surgery_counts = df['Surgery_Type'].value_counts()
print("\nSurgery Type Counts:")
print(surgery_counts)

# Average Recovery Time by Diagnosis (Top 10)
top_diagnoses_index = df['Diagnosis'].value_counts().head(10).index
avg_recovery = df[df['Diagnosis'].isin(top_diagnoses_index)].groupby('Diagnosis')['Recovery_Days'].mean().sort_values()
plt.figure(figsize=(10,5))
sns.barplot(x=avg_recovery.values, y=avg_recovery.index, palette='Blues_d')
plt.title("Average Recovery Time for Top 10 Diagnoses")
plt.xlabel("Average Recovery Days")
plt.show()

# Interactive bar chart for SAT by Doctor
avg_sat = df.groupby('Doctor_Name')['SAT'].mean().sort_values().reset_index()
fig = px.bar(avg_sat, x='Doctor_Name', y='SAT', title='Average SAT per Doctor')
fig.update_layout(xaxis={'categoryorder':'total ascending'})
fig.show()

# Interactive bar chart for Recovery Time by Hospital
recovery_by_hospital = df.groupby('Hospital_Name')['Recovery_Days'].mean().sort_values().reset_index()
fig = px.bar(recovery_by_hospital, x='Hospital_Name', y='Recovery_Days', title='Avg Recovery Time by Hospital', color='Recovery_Days')
fig.update_layout(xaxis={'categoryorder':'total ascending'})
fig.show()

# 1. Does Family History or Allergies Influence Diagnosis?
top_diagnoses = df['Diagnosis'].value_counts().head(5).index
df_top_diag = df[df['Diagnosis'].isin(top_diagnoses)]

# -- Family History vs Diagnosis
family_crosstab = pd.crosstab(df_top_diag['FamilyHistory'], df_top_diag['Diagnosis'])
print("\nFamily History vs Diagnosis:")
print(family_crosstab)

if not family_crosstab.empty:
    chi2, p_fam, _, _ = chi2_contingency(family_crosstab)
    print(f"Chi-square test (Family History): p = {p_fam:.4f}")
    family_crosstab.plot(kind='bar', stacked=True, figsize=(8,5), colormap='viridis')
    plt.title('Top Diagnoses by Family History')
    plt.xlabel('Family History')
    plt.ylabel('Count')
    plt.show()
else:
    print("No data available for Family History vs Diagnosis analysis.")

# -- Allergies vs Diagnosis
df_allergy_valid = df_top_diag[df_top_diag['Allergies'].notna() & (df_top_diag['Allergies'] != '')]
allergy_crosstab = pd.crosstab(df_allergy_valid['Allergies'], df_allergy_valid['Diagnosis'])
print("\nAllergies vs Diagnosis:")
print(allergy_crosstab)

if not allergy_crosstab.empty:
    chi2, p_allergy, _, _ = chi2_contingency(allergy_crosstab)
    print(f"Chi-square test (Allergies): p = {p_allergy:.4f}")
    allergy_crosstab.plot(kind='bar', stacked=True, figsize=(8,5), colormap='plasma')
    plt.title('Top Diagnoses by Allergy Status')
    plt.xlabel('Allergies')
    plt.ylabel('Count')
    plt.show()
else:
    print("No valid data found for Allergies vs Diagnosis. Skipping analysis.")

# Insight: If p < 0.05, it indicates a statistically significant relationship.

# 2. Do Some Surgeries Take Longer to Treat and Recover From?
surgery_avg = df.groupby('Surgery_Type')[['Treatment_Days', 'Recovery_Days']].mean().sort_values('Treatment_Days', ascending=False)
print("\nAverage Treatment and Recovery Days by Surgery Type:")
print(surgery_avg.head())

# Boxplot for Top 5 Surgery Types
top_surgeries = df['Surgery_Type'].value_counts().head(5).index
plt.figure(figsize=(10,6))
sns.boxplot(data=df[df['Surgery_Type'].isin(top_surgeries)], x='Surgery_Type', y='Treatment_Days', palette='Set2')
plt.title('Treatment Duration by Surgery Type')
plt.xlabel('Surgery Type')
plt.ylabel('Treatment Days')
plt.xticks(rotation=45)
plt.show()

# Treatment vs Recovery Correlation
df_corr = df[['Treatment_Days', 'Recovery_Days']].dropna()
corr, p_val = pearsonr(df_corr['Treatment_Days'], df_corr['Recovery_Days'])
print(f"\nCorrelation between Treatment and Recovery Days: r = {corr:.3f}, p = {p_val:.4f}")

sns.lmplot(data=df_corr, x='Treatment_Days', y='Recovery_Days', aspect=1.5)
plt.title('Relationship: Treatment vs Recovery Duration')
plt.xlabel('Treatment Days')
plt.ylabel('Recovery Days')
plt.show()

# 3. Does Hospital Affect Recovery Time for the Most Common Diagnosis?
top_diagnosis = df['Diagnosis'].value_counts().idxmax()
print(f"\nMost Common Diagnosis: {top_diagnosis}")
df_diag_hosp = df[df['Diagnosis'] == top_diagnosis]
top_hospitals = df_diag_hosp['Hospital_Name'].value_counts().head(5).index

plt.figure(figsize=(12,6))
sns.boxplot(data=df_diag_hosp[df_diag_hosp['Hospital_Name'].isin(top_hospitals)],
            x='Hospital_Name', y='Recovery_Days', palette='Set3')
plt.title(f'Recovery Days Across Hospitals for: {top_diagnosis}')
plt.xlabel('Hospital Name')
plt.ylabel('Recovery Days')
plt.xticks(rotation=45)
plt.show()

# 4. Are Younger Patients Recovering Faster?
df_age_rec = df[['Age', 'Recovery_Days']].dropna()
corr_age, p_age = pearsonr(df_age_rec['Age'], df_age_rec['Recovery_Days'])
print(f"\nCorrelation between Age and Recovery Days: r = {corr_age:.3f}, p = {p_age:.4f}")

sns.lmplot(data=df_age_rec, x='Age', y='Recovery_Days', height=5, aspect=1.5)
plt.title('Does Age Impact Recovery Time?')
plt.xlabel('Age')
plt.ylabel('Recovery Days')
plt.show()

# 5. Is There a Gender Difference in Recovery Time?
gender_crosstab = pd.crosstab(df['Gender'], df['Diagnosis'])
print("\nGender Distribution Across Diagnoses:")
print(gender_crosstab)

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Gender', y='Recovery_Days', palette='coolwarm')
plt.title('Recovery Days by Gender')
plt.xlabel('Gender')
plt.ylabel('Recovery Days')
plt.show()


# SENTIMENT ANALYSIS
df['Polarity'] = df['Feedback'].apply(lambda text: TextBlob(str(text)).sentiment.polarity)
df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0 else 'Negative' if p < 0 else 'Neutral')

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sentiment')
plt.title('Sentiment Distribution of Patient Feedback')
plt.xlabel('Sentiment')
plt.ylabel('Number of Feedbacks')
plt.show()

# Preprocess the data
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model
print("Random Forest Classifier Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
xgb_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
print("XGBoost Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Define the hyperparameter tuning space
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [50, 100, 200],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(xgb.XGBClassifier(objective='multi:softmax', num_class=3), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding accuracy
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

def predict_diagnosis(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    input_df['Gender'] = label_encoders['Gender'].transform(input_df['Gender'])
    input_df['Medication'] = label_encoders['Medication'].transform(input_df['Medication'])
    
    # Make predictions
    diagnosis_prediction = xgb_model.predict(input_df)
    
    # Decode the diagnosis
    diagnosis_decoded = label_encoders['Diagnosis'].inverse_transform(diagnosis_prediction)
    
    return diagnosis_decoded[0]

# Example user input
user_input = {
    'Age': ,
    'Gender': 'Male',
    'Heart_Rate': 85,
    'Temperature': 99.0,
    'Systolic': 140,
    'Diastolic': 90,
    'X-ray_Result': 'Normal',  # Assuming you have a way to encode this
    'Lab_Test_Results': 'Normal',  # Assuming you have a way to encode this
    'Medication': 'Metformin'
}

# Make prediction
predicted_diagnosis = predict_diagnosis(user_input)
print(f"Predicted Diagnosis: {predicted_diagnosis}")


