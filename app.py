import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendationV2.csv")
    return df

# -------------------------
# Load pre-trained model
# -------------------------
@st.cache_data
def load_model():
    model = joblib.load("crop_model.pkl")
    return model

# -------------------------
# Prepare scaler and encoder from dataset
# -------------------------
@st.cache_data
def prepare_scaler_encoder(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    encoder = LabelEncoder()
    encoder.fit(y)

    scaler = StandardScaler()
    scaler.fit(X)

    return scaler, encoder, X.columns  # Also return feature columns

# -------------------------
# Crop Rotation Predictor
# -------------------------
def crop_rotation_predictor(crop_name, df, model, scaler, encoder, feature_columns, top_k=5):
    crop_data = df[df["label"] == crop_name]

    if crop_data.empty:
        return []

    # Reindex to dataset feature columns
    crop_data = crop_data.reindex(columns=feature_columns, fill_value=0)

    # --- Fix: Align with model’s expected features ---
    if hasattr(model, "n_features_in_"):
        expected = model.n_features_in_
        actual = crop_data.shape[1]

        if actual < expected:
            # Add dummy zero columns for missing features
            for i in range(expected - actual):
                crop_data[f"missing_{i}"] = 0
        elif actual > expected:
            # Trim extras
            crop_data = crop_data.iloc[:, :expected]

    # Average growing conditions
    avg_conditions = crop_data.mean().to_frame().T
    avg_scaled = scaler.transform(avg_conditions)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(avg_scaled)[0]
        top_indices = probs.argsort()[::-1]
        top_crops = encoder.inverse_transform(top_indices)
        filtered = [c for c in top_crops if c != crop_name]
        return filtered[:top_k]
    else:
        return []

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.title("🌱 Crop Rotation & Recommendation System")

    df = load_data()
    model = load_model()
    scaler, encoder, feature_columns = prepare_scaler_encoder(df)

    st.sidebar.header("Options")
    crop_list = sorted(df["label"].unique())
    selected_crop = st.sidebar.selectbox("Select your current crop:", crop_list)

    if st.sidebar.button("Recommend Rotations"):
        recommendations = crop_rotation_predictor(
            selected_crop, df, model, scaler, encoder, feature_columns, top_k=5
        )
        st.write(f"✅ Given Crop: **{selected_crop}**")
        if recommendations:
            st.write("🌾 Recommended Next Crops for Rotation:")
            for i, crop in enumerate(recommendations, 1):
                st.write(f"{i}. {crop}")
        else:
            st.write("⚠️ No recommendations available (model may not support probabilities).")

if __name__ == "__main__":
    main()
