import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("processed_dataset.csv")

# Split into features and target
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Encode target labels (convert disease names into numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder for later use in prediction
joblib.dump(label_encoder, "label_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ XGBoost Model Accuracy: {accuracy:.2f}")

# Save trained model
joblib.dump(model, "disease_prediction_model_tuned.pkl")
print("✅ XGBoost Model saved successfully!")
