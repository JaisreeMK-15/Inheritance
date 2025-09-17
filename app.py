import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder # <-- ADD THIS LINE

# --- Load All Pre-trained Artifacts ---
@st.cache_resource
def load_artifacts():
    """Load the pre-trained model, scaler, encoder, and feature columns."""
    model = joblib.load("crop_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    feature_columns = joblib.load("feature_columns.joblib")
    return model, scaler, encoder, feature_columns

@st.cache_data
def load_data():
    """Load the raw CSV data for reference."""
    df = pd.read_csv("Crop_recommendationV2.csv")   
    return df

# --- Feature Engineering Function (to apply to user input) ---
def engineer_features_for_prediction(df):
    """Applies the same feature engineering steps to the input data."""
    df_eng = df.copy()

    # Apply the same transformations as in the training script
    df_eng['Temperature-Humidity Index'] = df_eng['temperature'] - (
        (0.55 - 0.0055 * df_eng['humidity']) * (df_eng['temperature'] - 14.5)
    )
    df_eng['Nutrient Balance Ratio'] = df_eng['N'] / (df_eng['P'] + df_eng['K'] + 1)
    df_eng['Water Availability Index'] = 0.7 * df_eng['soil_moisture'] + 0.3 * df_eng['rainfall']
    
    # Photosynthesis Potential needs to be handled carefully as min/max from training isn't available
    # For a single row, we can approximate or set a default. Here, we'll just compute the raw value.
    # A more robust solution might save training min/max values.
    raw_pp = df_eng['sunlight_exposure'] * df_eng['co2_concentration'] * df_eng['temperature']
    # We will assume a default normalized value, as scaling a single value is not meaningful.
    df_eng['Photosynthesis Potential'] = 0.5 # Default or simplified value

    df_eng['Soil Fertility Index'] = df_eng['organic_matter'] * ((df_eng['N'] + df_eng['P'] + df_eng['K']) / 3)

    # Categorical Features
    df_eng['Nutrient_Level'] = pd.cut(df_eng['N'] + df_eng['P'] + df_eng['K'], bins=[0, 150, 300, 450], labels=['Low', 'Moderate', 'High'])
    df_eng['Soil_Moisture_Class'] = pd.cut(df_eng['soil_moisture'], bins=[0, 15, 30, 100], labels=['Dry', 'Optimal', 'Wet'])
    df_eng['Frost_Risk_Level'] = pd.cut(df_eng['frost_risk'], bins=[-1, 33, 66, 100], labels=['Low', 'Medium', 'High'])
    df_eng['Urban_Proximity_Class'] = pd.cut(df_eng['urban_area_proximity'], bins=[-1, 10, 30, np.inf], labels=['Near', 'Moderate', 'Remote'])
    df_eng['Irrigation_Category'] = pd.cut(df_eng['irrigation_frequency'], bins=[-1, 1, 3, 7], labels=['Low', 'Moderate', 'High'])

    # Encoding
    ordinal_features = ['Nutrient_Level', 'Soil_Moisture_Class', 'Frost_Risk_Level', 'Irrigation_Category']
    for col in ordinal_features:
        le = LabelEncoder()
        df_eng[col] = le.fit_transform(df_eng[col])

    df_eng = pd.get_dummies(df_eng, columns=['Urban_Proximity_Class'])

    return df_eng

# --- Crop Rotation Predictor ---
def crop_rotation_predictor(crop_name, df, model, scaler, encoder, feature_columns, top_k=5):
    # Get the average conditions for the selected crop from the original dataset
    crop_data = df[df["label"] == crop_name].drop("label", axis=1)

    if crop_data.empty:
        return []

    # Get the mean of the raw features
    avg_conditions_raw = pd.DataFrame(crop_data.mean()).T
    
    # Apply feature engineering to this single row of average conditions
    avg_conditions_engineered = engineer_features_for_prediction(avg_conditions_raw)

    # Align columns with the training data, adding missing ones and removing extras
    # This is the key step to prevent the feature mismatch error
    final_df = avg_conditions_engineered.reindex(columns=feature_columns, fill_value=0)

    # Scale the final data
    avg_scaled = scaler.transform(final_df)

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(avg_scaled)[0]
        top_indices = probs.argsort()[::-1]
        top_crops = encoder.inverse_transform(top_indices)
        
        # Filter out the input crop itself
        filtered = [c for c in top_crops if c.lower() != crop_name.lower()]
        return filtered[:top_k]
    else:
        return []

# --- Streamlit App ---
def main():
    st.title("🌱 Crop Rotation & Recommendation System")

    try:
        model, scaler, encoder, feature_columns = load_artifacts()
        df = load_data()
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run the train_model.py script first.")
        return

    st.sidebar.header("Options")
    crop_list = sorted(df["label"].unique())
    selected_crop = st.sidebar.selectbox("Select your current crop:", crop_list)

    if st.sidebar.button("Recommend Rotations"):
        with st.spinner('Calculating recommendations...'):
            recommendations = crop_rotation_predictor(
                selected_crop, df, model, scaler, encoder, feature_columns, top_k=5
            )
        
        st.write(f"✅ Given Crop: *{selected_crop}*")
        if recommendations:
            st.write("🌾 Recommended Next Crops for Rotation:")
            for i, crop in enumerate(recommendations, 1):
                st.write(f"{i}. {crop}")
        else:
            st.write("⚠ No recommendations available for this crop.")

if __name__ == "__main__":
    main()