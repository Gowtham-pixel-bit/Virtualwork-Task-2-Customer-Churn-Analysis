import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("data/customer_churn.csv")

print("========== ORIGINAL DATA ==========")
print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

# Remove Extra Spaces
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Replace Empty Values
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

print("\n========== MISSING VALUES BEFORE CLEANING ==========")
print(df.isnull().sum())

# Convert TotalCharges into Numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove Duplicate Records
df.drop_duplicates(inplace=True)

# Handle Missing Values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Standardize Text Columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Validate Customer ID
df = df.dropna(subset=["customerID"])

# Save Cleaned Dataset
df.to_csv("output/cleaned_data.csv", index=False)

print("\n========== CLEANING RESULTS ==========")
print("Total Records After Cleaning:", len(df))
print("Total Columns:", len(df.columns))

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Records After Cleaning:")
print(df.duplicated().sum())

print("\nCleaned dataset saved successfully!")
# CUSTOMER CHURN ANALYSIS

print("\n========== CUSTOMER CHURN ANALYSIS ==========")

# Total Customers
total_customers = len(df)

# Churned Customers
churned = df[df["Churn"] == "Yes"]

# Retained Customers
retained = df[df["Churn"] == "No"]

# Calculate Churn Rate
churn_rate = (len(churned) / total_customers) * 100

print("Total Customers:", total_customers)
print("Churned Customers:", len(churned))
print("Retained Customers:", len(retained))
print("Customer Churn Rate:", round(churn_rate, 2), "%")

# Gender Wise Churn
print("\n========== GENDER WISE CHURN ==========")
print(pd.crosstab(df["gender"], df["Churn"]))

# Contract Wise Churn
print("\n========== CONTRACT WISE CHURN ==========")
print(pd.crosstab(df["Contract"], df["Churn"]))

# Payment Method Analysis
print("\n========== PAYMENT METHOD ANALYSIS ==========")
print(pd.crosstab(df["PaymentMethod"], df["Churn"]))

# Monthly Charges Analysis
print("\n========== MONTHLY CHARGES ==========")
print(df.groupby("Churn")["MonthlyCharges"].mean())

# Tenure Analysis
print("\n========== CUSTOMER TENURE ==========")
print(df.groupby("Churn")["tenure"].mean())

print("\nCustomer churn analysis completed!")
# CUSTOMER CHURN VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("output", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Customer Churn Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.savefig("output/churn_distribution.png")
plt.close()

# 2. Gender Wise Churn
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="gender", hue="Churn")
plt.title("Gender Wise Customer Churn")
plt.savefig("output/gender_churn.png")
plt.close()

# 3. Contract Wise Churn
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Wise Customer Churn")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("output/contract_churn.png")
plt.close()

# 4. Payment Method Analysis
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="PaymentMethod", hue="Churn")
plt.title("Payment Method and Customer Churn")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("output/payment_churn.png")
plt.close()

# 5. Monthly Charges Analysis
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.savefig("output/monthly_charges.png")
plt.close()

print("\nAll Visualization Graphs Generated Successfully!")
# CUSTOMER CHURN SUMMARY REPORT

with open("output/churn_report.txt", "w") as file:

    file.write("CUSTOMER CHURN ANALYSIS REPORT\n")
    file.write("==============================\n\n")

    file.write(f"Total Customers: {len(df)}\n")
    file.write(f"Churned Customers: {len(churned)}\n")
    file.write(f"Retained Customers: {len(retained)}\n")
    file.write(f"Customer Churn Rate: {round(churn_rate, 2)}%\n\n")

    file.write("KEY FINDINGS\n")
    file.write("------------\n")

    contract_churn = pd.crosstab(df["Contract"], df["Churn"])

    file.write("\nContract-wise Churn:\n")
    file.write(contract_churn.to_string())

    file.write("\n\nPayment Method-wise Churn:\n")
    file.write(pd.crosstab(df["PaymentMethod"], df["Churn"]).to_string())

    file.write("\n\nAverage Monthly Charges:\n")
    file.write(df.groupby("Churn")["MonthlyCharges"].mean().to_string())

print("Churn report generated successfully!")