import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

print("Starting the training process...")

# --- 1. Load and Clean Data ---
print("Step 1: Loading and cleaning data...")
try:
    # Load the original dataset
    crop_df = pd.read_csv("Crop_recommendationV2.csv")

    # Anomaly detection using Isolation Forest (as in the PDF)
    features_for_anomaly = crop_df.drop(columns=['label'])
    iso = IsolationForest(contamination=0.02, random_state=42)
    crop_df['anomaly'] = iso.fit_predict(features_for_anomaly)

    # Create the cleaned dataframe by removing anomalies
    clean_crop_df = crop_df[crop_df['anomaly'] == 1].copy()
    clean_crop_df = clean_crop_df.drop(columns=['anomaly'])
    print(f"Data cleaned. Shape is now: {clean_crop_df.shape}")

except FileNotFoundError:
    print("Error: Crop_recommendationV2.csv not found. Make sure it's in the same directory.")
    exit()


# --- 2. Feature Engineering ---
print("Step 2: Performing feature engineering...")
# This section replicates the feature creation from your PDF
# Derived Feature 1: Temperature-Humidity Index (THI)
clean_crop_df['Temperature-Humidity Index'] = clean_crop_df['temperature'] - (
    (0.55 - 0.0055 * clean_crop_df['humidity']) * (clean_crop_df['temperature'] - 14.5)
)
# Derived Feature 2: Nutrient Balance Ratio (NBR)
clean_crop_df['Nutrient Balance Ratio'] = clean_crop_df['N'] / (clean_crop_df['P'] + clean_crop_df['K'] + 1)
# Derived Feature 3: Water Availability Index (WAI)
clean_crop_df['Water Availability Index'] = 0.7 * clean_crop_df['soil_moisture'] + 0.3 * clean_crop_df['rainfall']
# Derived Feature 4: Photosynthesis Potential (PP)
raw_pp = clean_crop_df['sunlight_exposure'] * clean_crop_df['co2_concentration'] * clean_crop_df['temperature']
clean_crop_df['Photosynthesis Potential'] = (raw_pp - raw_pp.min()) / (raw_pp.max() - raw_pp.min())
# Derived Feature 5: Soil Fertility Index (SFI)
clean_crop_df['Soil Fertility Index'] = clean_crop_df['organic_matter'] * ((clean_crop_df['N'] + clean_crop_df['P'] + clean_crop_df['K']) / 3)

# Categorical Features
clean_crop_df['Nutrient_Level'] = pd.cut(clean_crop_df['N'] + clean_crop_df['P'] + clean_crop_df['K'], bins=[0, 150, 300, 450], labels=['Low', 'Moderate', 'High'])
clean_crop_df['Soil_Moisture_Class'] = pd.cut(clean_crop_df['soil_moisture'], bins=[0, 15, 30, 100], labels=['Dry', 'Optimal', 'Wet'])
clean_crop_df['Frost_Risk_Level'] = pd.cut(clean_crop_df['frost_risk'], bins=[-1, 33, 66, 100], labels=['Low', 'Medium', 'High'])
clean_crop_df['Urban_Proximity_Class'] = pd.cut(clean_crop_df['urban_area_proximity'], bins=[-1, 10, 30, np.inf], labels=['Near', 'Moderate', 'Remote'])
clean_crop_df['Irrigation_Category'] = pd.cut(clean_crop_df['irrigation_frequency'], bins=[-1, 1, 3, 7], labels=['Low', 'Moderate', 'High'])

print("Feature engineering complete.")


# --- 3. Encoding Categorical Features ---
print("Step 3: Encoding categorical features...")
# Define which features are ordinal and nominal
ordinal_features = ['Nutrient_Level', 'Soil_Moisture_Class', 'Frost_Risk_Level', 'Irrigation_Category']
nominal_features = ['Urban_Proximity_Class']

# Label Encoding for ordinal features
for col in ordinal_features:
    le = LabelEncoder()
    clean_crop_df[col] = le.fit_transform(clean_crop_df[col])

# One-Hot Encoding for nominal features
clean_crop_df = pd.get_dummies(clean_crop_df, columns=nominal_features)
print("Encoding complete.")


# --- 4. Prepare Data for Model Training ---
print("Step 4: Preparing data for the model...")
# Define features (X) and target (y)
X = clean_crop_df.drop('label', axis=1)
y = clean_crop_df['label']

# Save the final list of feature columns
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.joblib')
print(f"Final feature count: {len(feature_columns)}. This should match the error message.")


# Encode the target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data is scaled and split.")


# --- 5. Train the Model ---
print("Step 5: Training the Random Forest model...")
# Using Random Forest as it was the best model in your analysis
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"Model training complete. Accuracy: {accuracy:.4f}")


# --- 6. Save Artifacts ---
print("Step 6: Saving model, scaler, and encoder...")
joblib.dump(model, "crop_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("\n🎉 Training process finished successfully! All files have been saved.")