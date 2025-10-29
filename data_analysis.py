import pandas as pd

# Load dataset
file_path = r"C:\Users\suraktim choudhury\Desktop\Smart Traffic Violation Pattern Detection\Indian_Traffic_Violations.csv"

# Step 1: Read CSV
df = pd.read_csv(file_path)

# Step 2: Basic Info
print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

# Step 3: Sample preview
print("\nFirst 5 Rows:\n", df.head())

# Step 4: Clean Data
df.dropna(inplace=True)  # remove rows with missing data
df.drop_duplicates(inplace=True)  # remove duplicates
df['Violation Type'] = df['Violation Type'].str.strip().str.lower()  # normalize text

print("\nCleaned Dataset Shape:", df.shape)

# Step 5: Simple Analysis
print("\nTop 5 Violation Types:")
print(df['Violation Type'].value_counts().head())

print("\nAverage Fine by Violation Type:")
print(df.groupby('Violation Type')['Fine Amount'].mean().sort_values(ascending=False).head())

