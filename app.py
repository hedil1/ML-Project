import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sklearn
import xgboost
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Smart PN Dashboard", layout="wide")
st.title(" Smart Dashboard - Zones Dangereuses 🇹🇳")

# =========================
# SESSION
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

    df["Latitude"] = df["Gouvernorat"].map(lambda x: noise(coords.get(x, (34.5, 9.5))[0]))
    df["Longitude"] = df["Gouvernorat"].map(lambda x: noise(coords.get(x, (34.5, 9.5))[1]))

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").fillna(34.5)
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").fillna(9.5)

    df["Nbre d'intersection"] = pd.to_numeric(
        df.get("Nbre d'intersection", 0), errors="coerce"
    ).fillna(1)

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

st.sidebar.success(f"Modèle actif: {model_name}")

# =========================
# STATS
# =========================


# =========================
# FILTER
# =========================
arr = st.sidebar.selectbox("Arrondissement", df["Arrondissement"].unique())
mois_filter = st.sidebar.selectbox("Mois", df["Mois"].unique())

df_filtered = df[(df["Arrondissement"] == arr) & (df["Mois"] == mois_filter)]

# =========================
# MAP
# =========================


# =========================
# HEATMAP
# =========================
st.subheader("🔥 Heatmap")

fig_heat = px.density_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    z="Dangereux",
    radius=25,
    center=dict(lat=34.5, lon=9.5),
    zoom=5,
    mapbox_style="open-street-map"
)

st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# EVOLUTION
# =========================
st.subheader("📈 Évolution")

df_evol = df.groupby("Mois")["Dangereux"].mean().reset_index()

fig_evol = px.line(df_evol, x="Mois", y="Dangereux", markers=True)
st.plotly_chart(fig_evol, use_container_width=True)

# =========================
# PREDICTION
# =========================
st.subheader("🔮 Prédiction")

zone = st.selectbox("Zone", df["Zone"].unique())
gouv = st.selectbox("Gouvernorat", df["Gouvernorat"].unique())
mois = st.selectbox("Mois", df["Mois"].unique())

securite = st.slider("Sécurité", 0, 10, 5)
intersections = st.number_input("Nbre intersections", 0, 50, 5)

input_data = pd.DataFrame([{
    "Zone": zone,
    "Gouvernorat": gouv,
    "Mois": mois,
    "Sécurité": securite,
    "Nbre d'intersection": intersections
}])

if st.button("🚀 Predict"):

    pred = model.predict(input_data)[0]

    proba = 0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0].max()

    lat, lon = coords.get(gouv, (34.5, 9.5))

    st.session_state.pred_points.append({
        "Latitude": lat,
        "Longitude": lon,
        "Prediction": "Danger" if pred == 1 else "Safe",
        "Arrondissement": f"{zone}_{gouv}",
        "Nbre d'intersection": intersections
    })

    if pred == 1:
        st.error("🔴 Zone Dangereuse")
    else:
        st.success("🟢 Zone Sûre")

    st.metric("Confiance", f"{proba*100:.2f}%")

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📉 Confusion Matrix")

X = df[feature_cols]
y = df["Dangereux"]

y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# =========================
# ROC CURVE
# =========================
st.subheader("📊 ROC Curve")

if hasattr(model, "predict_proba"):

    y_proba = model.predict_proba(X)[:,1]

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.legend()
    st.pyplot(fig)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📌 Importance des variables")

try:
    importances = model.named_steps["clf"].feature_importances_
    features = model.named_steps["prep"].get_feature_names_out()

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(df_imp.set_index("Feature"))
except:
    st.info("Importance non disponible pour ce modèle")