import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("processed_dataset.csv")

# Check column names
print("Columns in dataset:", df.columns)

# Encode the 'prognosis' column (target variable)
label_encoder = LabelEncoder()
df["prognosis"] = label_encoder.fit_transform(df["prognosis"])

# Save the encoder for future use
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

# Save the preprocessed dataset
df.to_csv("preprocessed_test_dataset.csv", index=False)
print("✅ Preprocessed dataset saved as 'preprocessed_test_dataset.csv'")
