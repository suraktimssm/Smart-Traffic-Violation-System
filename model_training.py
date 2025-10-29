import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = r"C:\Users\suraktim choudhury\Desktop\Smart Traffic Violation Pattern Detection\Indian_Traffic_Violations.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

# Label encode categorical columns
label_cols = ['Violation Type', 'Vehicle Type', 'Weather', 'Road Status']
le = LabelEncoder()

for col in label_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Define features and target
X = df[['Vehicle Type', 'Weather', 'Road Status', 'Fine Amount']]
y = df['Violation Type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
