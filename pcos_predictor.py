import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

MODEL_FILENAME = 'pcos_model.pkl'

# Step 1: Train and save the model if not already saved
if not os.path.exists(MODEL_FILENAME):
    print("🔁 Training model for the first time...")

    # Load the dataset
    df = pd.read_csv('pcos_data.csv')

    # Select features and target
    X = df[['Age', 'BMI', 'Hair_Growth', 'Cycle_Length', 'Acne', 'Weight_Gain']]
    y = df['PCOS']

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_FILENAME)

    # Show accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained and saved! Accuracy: {accuracy * 100:.2f}%\n")
else:
    print("✅ Model already trained. Loading from file...\n")

# Step 2: Load the model
try:
    model = joblib.load(MODEL_FILENAME)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# Step 3: Take user input
print("👩‍⚕️ Welcome to CheckLyfAI - PCOS Predictor")
print("Please enter the following details:")

try:
    age = int(input("Age (in years): "))
    bmi = float(input("BMI (Body Mass Index): "))
    hair_growth = int(input("Unusual Hair Growth? (1 = Yes, 0 = No): "))
    cycle_length = int(input("Average Menstrual Cycle Length (in days): "))
    acne = int(input("Acne? (1 = Yes, 0 = No): "))
    weight_gain = int(input("Sudden Weight Gain? (1 = Yes, 0 = No): "))

    # Step 4: Predict
    user_input = [[age, bmi, hair_growth, cycle_length, acne, weight_gain]]
    prediction = model.predict(user_input)

    print("\n🧾 Result:")
    if prediction[0] == 1:
        print("⚠️ You may have symptoms related to PCOS. Please consult a healthcare provider.")
    else:
        print("✅ You are not showing strong signs of PCOS based on this input.")

except ValueError:
    print("❌ Invalid input. Please enter numbers only.")

