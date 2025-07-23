import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('pcos_data.csv')

# Drop missing values (if any)
df = df.dropna()

# Features and label
X = df.drop('PCOS', axis=1)
y = df['PCOS']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open('pcos_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as pcos_model.pkl")

