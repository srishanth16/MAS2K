import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load preprocessed test dataset
df_test = pd.read_csv("preprocessed_test_dataset.csv")

# Separate features and target
X_test = df_test.drop(columns=["prognosis"])
y_test = df_test["prognosis"]

# Load trained model
model = joblib.load("disease_prediction_model_tuned.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Evaluation Accuracy: {accuracy:.2f}")
 
