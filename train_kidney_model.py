import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
data = pd.read_csv('kidney_disease.csv')

# Preprocessing: drop id, drop rows with missing values
X = data.drop(['id', 'classification'], axis=1)
y = data['classification']

# Encode categorical columns
for col in X.columns:
    if X[col].dtype == 'object' or X[col].isnull().any():
        X[col] = X[col].astype('category').cat.codes

# Encode target
if y.dtype == 'object':
    y = y.astype('category').cat.codes

# Drop rows with missing values after encoding
X = X.dropna()
y = y.loc[X.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open('random_forest_kidney_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model trained and saved as random_forest_kidney_model.pkl')
