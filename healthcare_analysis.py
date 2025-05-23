import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

#Loading the dataset
df = pd.read_csv('healthcare_dataset.csv')

# Numeric columns - fill missing with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns - fill missing with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Print the first 15 rows
df.head(15)

# Handling Outkiners - Uing IQR
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

# NEw cols with the converted Time
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

# New col named 'Age_Group'
df['Age_Group'] = df['Age'].apply(age_group)
df.head(15)

# Check for non-numeric values in scale_cols
for col in scale_cols:
    non_numeric = df[col][~df[col].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))]
    if not non_numeric.empty:
        print(f"Non-numeric values in '{col}':")
        print(non_numeric.unique())

# Try to convert SAT to numeric, force errors to NaN
df['SAT'] = pd.to_numeric(df['SAT'], errors='coerce')

# Fill missing SAT scores with median (or mean)
df['SAT'].fillna(df['SAT'].median(), inplace=True)


scale_cols = ['Age', 'Heart_Rate', 'Temperature', 'SAT', 'Treatment_Days', 'Recovery_Days', 'Systolic', 'Diastolic']

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[scale_cols]), columns=[col + '_scaled' for col in scale_cols])
df = pd.concat([df, df_scaled], axis=1)

# Set default Seaborn theme
sns.set(style="whitegrid")

# Distribution of Age, Treatment Days, and SAT
df[['Age', 'Treatment_Days', 'Recovery_Days', 'SAT']].hist(figsize=(10, 6), bins=20)
plt.suptitle("Distributions of Age, Treatment & Recovery Days, and SAT")
plt.tight_layout()
plt.show()

# Average SAT by Age Group
print("\nAverage SAT Score by Age Group:")
print(df.groupby('Age_Group')['SAT'].mean())

# 3. Average Treatment & Recovery Days by Surgery Type
print("\nAverage Treatment and Recovery Days by Surgery Type:")
print(df.groupby('Surgery_Type')[['Treatment_Days', 'Recovery_Days']].mean())

# 4. Scatterplot: Treatment vs Recovery
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
plt.figure(figsize=(10,5))
top_diagnoses = df['Diagnosis'].value_counts().head(10)
sns.barplot(x=top_diagnoses.index, y=top_diagnoses.values, palette='coolwarm')
plt.title("Top 10 Diagnoses")
plt.xticks(rotation=45)
plt.ylabel("Number of Patients")
plt.show()

# Boxplot: SAT Score by Age Group
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='Age_Group', y='SAT', palette='Pastel1')
plt.title("SAT Scores by Age Group")
plt.ylabel("Satisfaction Score")
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10,6))
corr_matrix = df[['Age', 'Heart_Rate', 'Temperature', 'SAT', 'Treatment_Days', 'Recovery_Days', 'Systolic', 'Diastolic']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Count of Surgery Types
plt.figure(figsize=(10,5))
sns.countplot(data=df, y='Surgery_Type', order=df['Surgery_Type'].value_counts().index, palette='Set3')
plt.title("Count of Different Surgery Types")
plt.xlabel("Number of Surgeries")
plt.ylabel("Surgery Type")
plt.show()

# Average Recovery Time by Diagnosis (Top 10)
top_diagnoses_index = df['Diagnosis'].value_counts().head(10).index
avg_recovery = df[df['Diagnosis'].isin(top_diagnoses_index)].groupby('Diagnosis')['Recovery_Days'].mean().sort_values()
plt.figure(figsize=(10,5))
sns.barplot(x=avg_recovery.values, y=avg_recovery.index, palette='Blues_d')
plt.title("Average Recovery Time for Top 10 Diagnoses")
plt.xlabel("Average Recovery Days")
plt.show()

# Violin Plot: Recovery Time by Gender
plt.figure(figsize=(7,5))
sns.violinplot(data=df, x='Gender', y='Recovery_Days', palette='muted')
plt.title("Recovery Time Distribution by Gender")
plt.ylabel("Recovery Days")
plt.show()

# 1. Does Family History or Allergies influence Diagnosis?
top_diagnoses = df['Diagnosis'].value_counts().head(5).index

# Family History vs Diagnosis
family_table = pd.crosstab(df[df['Diagnosis'].isin(top_diagnoses)]['FamilyHistory'],
                           df[df['Diagnosis'].isin(top_diagnoses)]['Diagnosis'])
print("Family History vs Diagnosis")
print(family_table)

# Chi-square test
chi2, p, _, _ = chi2_contingency(family_table)
print(f"Chi-square test (Family History): p = {p:.4f}")

# Plot
family_table.plot(kind='bar', stacked=True, figsize=(8,5), colormap='viridis')
plt.title('Diagnosis by Family History')
plt.xlabel('Family History')
plt.ylabel('Count')
plt.show()

# Allergies vs Diagnosis
allergy_table = pd.crosstab(df[df['Diagnosis'].isin(top_diagnoses)]['Allergies'],
                            df[df['Diagnosis'].isin(top_diagnoses)]['Diagnosis'])
print("Allergies vs Diagnosis")
print(allergy_table)

chi2, p, _, _ = chi2_contingency(allergy_table)
print(f"Chi-square test (Allergies): p = {p:.4f}")

# Plot
allergy_table.plot(kind='bar', stacked=True, figsize=(8,5), colormap='plasma')
plt.title('Diagnosis by Allergy Status')
plt.xlabel('Allergies')
plt.ylabel('Count')
plt.show()

# 2. Do some surgeries take longer to treat and recover from?
surgery_avg = df.groupby('Surgery_Type')[['Treatment_Days', 'Recovery_Days']].mean().sort_values('Treatment_Days', ascending=False)
print("\nAverage Treatment and Recovery Days by Surgery Type:")
print(surgery_avg)

# Boxplot for Treatment Days (Top 5 Surgeries)
top_surgeries = df['Surgery_Type'].value_counts().head(5).index
plt.figure(figsize=(10,6))
sns.boxplot(data=df[df['Surgery_Type'].isin(top_surgeries)], x='Surgery_Type', y='Treatment_Days', palette='Set2')
plt.title('Treatment Days by Surgery Type')
plt.xlabel('Surgery Type')
plt.ylabel('Treatment Days')
plt.xticks(rotation=45)
plt.show()

# Correlation between Treatment and Recovery Days
corr, p_val = pearsonr(df['Treatment_Days'], df['Recovery_Days'])
print(f"Correlation between Treatment and Recovery Days: r = {corr:.3f}, p = {p_val:.4f}")

# Scatterplot with regression line
sns.lmplot(data=df, x='Treatment_Days', y='Recovery_Days', aspect=1.5)
plt.title('Treatment vs Recovery Duration')
plt.xlabel('Treatment Days')
plt.ylabel('Recovery Days')
plt.show()

# 3. Do hospitals affect recovery time for the most common diagnosis?
top_diagnosis = df['Diagnosis'].value_counts().idxmax()
print(f"Top Diagnosis: {top_diagnosis}")

# Filter data
df_top_diag = df[df['Diagnosis'] == top_diagnosis]
top_hospitals = df_top_diag['Hospital_Name'].value_counts().head(5).index

# Boxplot: Recovery Days by Hospital
plt.figure(figsize=(12,6))
sns.boxplot(data=df_top_diag[df_top_diag['Hospital_Name'].isin(top_hospitals)], 
            x='Hospital_Name', y='Recovery_Days', palette='Set3')
plt.title(f'Recovery Days by Hospital ({top_diagnosis})')
plt.xlabel('Hospital')
plt.ylabel('Recovery Days')
plt.xticks(rotation=45)
plt.show()

# ANOVA test
model = ols('Recovery_Days ~ C(Hospital_Name)', data=df_top_diag[df_top_diag['Hospital_Name'].isin(top_hospitals)]).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print("\nANOVA: Hospital Impact on Recovery Days")
print(anova_results)

# 4. How does Age Group affect Treatment and Recovery?
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Age_Group', y='Treatment_Days', palette='pastel')
plt.title('Treatment Days by Age Group')
plt.ylabel('Treatment Days')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Age_Group', y='Recovery_Days', palette='pastel')
plt.title('Recovery Days by Age Group')
plt.ylabel('Recovery Days')
plt.show()

# 5. Do Gender and Allergies affect Recovery?
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='Gender', y='Recovery_Days', hue='Allergies', palette='Set1')
plt.title('Recovery Days by Gender and Allergy Status')
plt.ylabel('Recovery Days')
plt.show()
