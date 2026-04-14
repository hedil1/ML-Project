import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="🚦 Smart PN Dashboard", layout="wide")
st.title("🚦 Smart Dashboard - Zones Dangereuses 🇹🇳")

# =========================
# SESSION STATE
# =========================
if "pred_points" not in st.session_state:
    st.session_state.pred_points = []

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("dataset_final.xlsx")
    df.columns = df.columns.str.strip()

    df["Dangereux"] = (df["Tués"] > 0).astype(int)
    df["Arrondissement"] = df["Zone"] + "_" + df["Gouvernorat"]

    coords = {
        "TN": (36.8, 10.18), "BR": (36.74, 10.21), "MN": (36.80, 10.09),
        "AR": (36.86, 10.19), "NB": (36.45, 10.73), "MS": (35.77, 10.82),
        "MH": (35.50, 11.06), "SS": (35.82, 10.60), "SF": (34.74, 10.76),
        "KR": (35.67, 10.09), "KS": (35.16, 8.83), "BZ": (37.27, 9.87),
        "GB": (33.88, 10.09), "MD": (33.35, 10.50), "ZG": (36.40, 10.14),
        "SL": (36.08, 9.37), "JN": (36.50, 8.78), "BJ": (36.72, 9.18),
        "SB": (35.03, 9.48), "KF": (36.18, 8.71), "GF": (34.42, 8.78)
    }

    np.random.seed(42)

    def noise(x):
        return x + np.random.uniform(-0.05, 0.05)

    df["Latitude"] = df["Gouvernorat"].map(
        lambda x: noise(coords.get(x, (34.5, 9.5))[0])
    )

    df["Longitude"] = df["Gouvernorat"].map(
        lambda x: noise(coords.get(x, (34.5, 9.5))[1])
    )

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").fillna(34.5)
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").fillna(9.5)

    df["Nbre d'intersection"] = pd.to_numeric(
        df.get("Nbre d'intersection", 0),
        errors="coerce"
    ).fillna(0)

    return df, coords

df, coords = load_data()

# =========================
# LOAD MODEL
# =========================
model_name = st.sidebar.selectbox(
    "Choisir modèle",
    ["GradientBoosting", "XGBoost", "RandomForest", "SVM", "KNN"]
)

model = joblib.load(f"models/{model_name}.pkl")
feature_cols = joblib.load("models/features.pkl")

# =========================
# STATS
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("Total", len(df))
col2.metric("Danger", df["Dangereux"].sum())
col3.metric("Safe", len(df) - df["Dangereux"].sum())

# =========================
# HEATMAP DYNAMIQUE
# =========================
st.subheader("🔥 Heatmap dynamique des zones dangereuses")

selected_month = st.slider(
    "Choisir le mois",
    int(df["Mois"].min()),
    int(df["Mois"].max()),
    int(df["Mois"].min())
)

df_month = df[df["Mois"] == selected_month]

fig_heat = px.density_mapbox(
    df_month,
    lat="Latitude",
    lon="Longitude",
    z="Dangereux",
    radius=20,
    center=dict(lat=34.5, lon=9.5),
    zoom=5,
    mapbox_style="open-street-map",
    title=f"Zones dangereuses - Mois {selected_month}"
)

st.plotly_chart(fig_heat, width='stretch')

# =========================
# TOP ZONES
# =========================
st.subheader("🚨 Top zones dangereuses")

top_zones = (
    df.groupby("Arrondissement")["Dangereux"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

st.table(top_zones)

# =========================
# PREDICTION
# =========================
st.subheader("🔮 Nouvelle prédiction")

col_a, col_b = st.columns(2)

with col_a:
    zone = st.selectbox("Zone", df["Zone"].unique())
    gouv = st.selectbox("Gouvernorat", df["Gouvernorat"].unique())
    mois = st.selectbox("Mois", df["Mois"].unique())

with col_b:
    securite = st.number_input("Sécurité (0-10)", 0, 10, 0)
    intersections = st.number_input("Nbre intersections", 0, 50, 0)

if st.button("🚀 Predict"):

    input_dict = {
        "Zone": zone,
        "Gouvernorat": gouv,
        "Mois": mois,
        "Sécurité": securite,
        "Nbre d'intersection": intersections
    }

    for col in feature_cols:
        if col not in input_dict:
            input_dict[col] = 0

    input_data = pd.DataFrame([input_dict])
    input_data = input_data[feature_cols]

    pred = model.predict(input_data)[0]

    proba = 0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0].max()

    if pred == 1:
        st.error("🔴 Zone Dangereuse")
    else:
        st.success("🟢 Zone Sûre")

    st.metric("Confiance", f"{proba*100:.2f}%")

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📊 Performance du modèle")

if st.button("Afficher Matrice de Confusion"):

    X = df[feature_cols]
    y = df["Dangereux"]

    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    st.pyplot(fig)