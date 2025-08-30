# -------------------------------
# Heart Disease Prediction Project
# -------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv('/content/drive/MyDrive/heart.csv')
print(df.head())

# -------------------------------
# Data exploration & visualization
# -------------------------------
plt.figure(figsize=(5, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 
    'slope', 'ca', 'thal'
]

for col in features:
    plt.figure(figsize=(5, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Check target distribution
print("Target value counts:\n", df['target'].value_counts())

# -------------------------------
# Outlier Detection & Removal (IQR Method)
# -------------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[((df < (lower_bound)) | (df > (upper_bound))).any(axis=1)]
print("Outliers detected:", len(outliers))

df_no_outliers = df[~((df < (lower_bound)) | (df > (upper_bound))).any(axis=1)]
print("Shape before:", df.shape)
print("Shape after removing outliers:", df_no_outliers.shape)

# -------------------------------
# Feature & target split
# -------------------------------
X = df_no_outliers.iloc[:, :-1]
y = df_no_outliers['target']

# -------------------------------
# Data Standardization
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=45
)

# -------------------------------
# Initialize Models
# -------------------------------
rf = RandomForestClassifier(n_estimators=300, random_state=45)
lr = LogisticRegression(max_iter=1000, random_state=45)
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=45)
svc = SVC()

models = {
    "RandomForest": rf,
    "LogisticRegression": lr,
    "KNN": knn,
    "DecisionTree": dt,
    "SVC": svc
}

# -------------------------------
# Train Models & Evaluate
# -------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\n{name} Train Accuracy: {train_acc:.4f}")
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    
    # Confusion Matrix
    y_pred = model.predict(X_test)
    print(f"Confusion Matrix ({name}):\n", confusion_matrix(y_test, y_pred))
    
    # Classification Report
    print(f"Classification Report ({name}):\n", classification_report(y_test, y_pred))

# -------------------------------
# Cross-Validation
# -------------------------------
for name, model in models.items():
    cv_score = cross_val_score(model, X_scaled, y, cv=5)
    print(f"\nCross Validation Score of {name}: {cv_score}")
    print(f"Mean CV Score of {name}: {cv_score.mean():.4f}")

# -------------------------------
# Test on New Data
# -------------------------------
new_data = [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]
new_data_scaled = scaler.transform([new_data])

print("\n--- Predictions on New Data ---")
for name, model in models.items():
    prediction = model.predict(new_data_scaled)
    result = "has heart disease" if prediction[0] == 1 else "does not have heart disease"
    print(f"{name} Prediction: The person {result}")

# -------------------------------
# Save Models
# -------------------------------
joblib.dump(rf, "rf_heart_disease_model.pkl")
joblib.dump(lr, "lr_heart_disease_model.pkl")
joblib.dump(knn, "knn_heart_disease_model.pkl")
joblib.dump(dt, "dt_heart_disease_model.pkl")
joblib.dump(svc, "svc_heart_disease_model.pkl")
print("\nAll models saved successfully.")