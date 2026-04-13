import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title=" Smart PN Dashboard", layout="wide")
st.title(" Smart Dashboard - Zones Dangereuses 🇹🇳")

# =========================
# SESSION STATE
# =========================
if "pred_points" not in st.session_state:
    st.session_state.pred_points = []

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("dataset_final.xlsx")
df.columns = df.columns.str.strip()

df["Dangereux"] = (df["Tués"] > 0).astype(int)

df["Arrondissement"] = df["Zone"] + "_" + df["Gouvernorat"]

# =========================
# GPS TUNISIE
# =========================
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

# =========================
# CLEAN DATA (IMPORTANT FIX)
# =========================
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").fillna(34.5)
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").fillna(9.5)

df["Nbre d'intersection"] = pd.to_numeric(
    df.get("Nbre d'intersection", 0),
    errors="coerce"
).fillna(0)

# =========================
# CLUSTERING
# =========================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[["Latitude", "Longitude"]])

# =========================
# LOAD MODEL (AVEC GESTION D'ERREUR ET RÉENTRAÎNEMENT)
# =========================
model_name = st.sidebar.selectbox(
    "Choisir modèle",
    ["GradientBoosting", "XGBoost", "RandomForest", "SVM", "KNN"]
)

model_path = f"models/{model_name}.pkl"

def prepare_features(dataframe):
    """Prépare les features pour l'entraînement"""
    df_model = dataframe.copy()
    
    # One-hot encoding pour les variables catégorielles
    categorical_cols = ["Zone", "Gouvernorat", "Mois"]
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    # Définir les colonnes de features (exclure les targets et metadata)
    exclude_cols = ["Tués", "Blessés", "Dangereux", "Arrondissement", 
                   "Cluster", "Latitude", "Longitude"]
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    return df_encoded, feature_cols

def train_and_save_models(df_encoded, feature_cols):
    """Entraîne et sauvegarde tous les modèles"""
    st.info("🔄 Première exécution : entraînement des modèles en cours...")
    
    X = df_encoded[feature_cols]
    y = df_encoded["Dangereux"]
    
    models = {
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)
    
    progress_bar = st.progress(0)
    for idx, (name, mdl) in enumerate(models.items()):
        mdl.fit(X, y)
        joblib.dump(mdl, f"models/{name}.pkl")
        progress_bar.progress((idx + 1) / len(models))
    
    # Sauvegarder les noms des features
    joblib.dump(feature_cols, "models/features.pkl")
    
    st.success("✅ Tous les modèles ont été entraînés avec succès !")
    progress_bar.empty()

# Vérifier si les modèles existent et sont valides
need_training = False

if not os.path.exists(model_path) or not os.path.exists("models/features.pkl"):
    need_training = True
else:
    # Essayer de charger pour vérifier la compatibilité
    try:
        _test_model = joblib.load(model_path)
        _test_features = joblib.load("models/features.pkl")
    except (AttributeError, Exception) as e:
        st.warning(f"⚠️ Modèles existants incompatibles : {e}")
        need_training = True

if need_training:
    df_encoded, feature_cols = prepare_features(df)
    train_and_save_models(df_encoded, feature_cols)

# Charger le modèle sélectionné et les features
try:
    model = joblib.load(model_path)
    feature_cols = joblib.load("models/features.pkl")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# =========================
# STATS
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("Total", len(df))
col2.metric("Danger", df["Dangereux"].sum())
col3.metric("Safe", len(df) - df["Dangereux"].sum())

# =========================
# FILTERS
# =========================
st.sidebar.header("Filtres")

arr = st.sidebar.selectbox("Arrondissement", df["Arrondissement"].unique())
mois_filter = st.sidebar.selectbox("Mois", df["Mois"].unique())

df_filtered = df[
    (df["Arrondissement"] == arr) &
    (df["Mois"] == mois_filter)
]

# =========================
# MAP
# =========================
st.subheader("🗺️ Carte intelligente")

map_df = df_filtered.copy()

# add predictions
if len(st.session_state.pred_points) > 0:
    pred_df = pd.DataFrame(st.session_state.pred_points)
    map_df = pd.concat([map_df, pred_df], ignore_index=True)

map_df["Color"] = map_df.get("Dangereux", 0)

if "Prediction" in map_df.columns:
    map_df.loc[map_df["Prediction"] == "Danger", "Color"] = 1
    map_df.loc[map_df["Prediction"] == "Safe", "Color"] = 0

fig_map = px.scatter_mapbox(
    map_df,
    lat="Latitude",
    lon="Longitude",
    color="Color",
    size="Nbre d'intersection",
    hover_name="Arrondissement",
    zoom=6,
    height=500,
    animation_frame="Mois"
)

fig_map.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_map, use_container_width=True)

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
    mapbox_style="open-street-map",
    animation_frame="Mois"
)

st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# PREDICTION
# =========================
st.subheader("🔮 Nouvelle prédiction")

zone = st.selectbox("Zone", df["Zone"].unique())
gouv = st.selectbox("Gouvernorat", df["Gouvernorat"].unique())
mois = st.selectbox("Mois", df["Mois"].unique())

securite = st.number_input("Sécurité", 0, 10, 0)
intersections = st.number_input("Nbre intersections", 0, 50, 0)

# Préparer les données d'entrée avec le même encodage que l'entraînement
input_data_dict = {
    "Zone": zone,
    "Gouvernorat": gouv,
    "Mois": mois,
    "Sécurité": securite,
    "Nbre d'intersection": intersections
}

# Créer un DataFrame avec une ligne et encoder comme lors de l'entraînement
input_df = pd.DataFrame([input_data_dict])
input_encoded = pd.get_dummies(input_df, columns=["Zone", "Gouvernorat", "Mois"], drop_first=True)

# Aligner avec les features d'entraînement
input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

if st.button("🚀 Predict"):

    pred = model.predict(input_encoded)[0]

    proba = 0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_encoded)[0].max()

    lat, lon = coords.get(gouv, (34.5, 9.5))
    lat += np.random.uniform(-0.03, 0.03)
    lon += np.random.uniform(-0.03, 0.03)

    st.session_state.pred_points.append({
        "Latitude": lat,
        "Longitude": lon,
        "Prediction": "Danger" if pred == 1 else "Safe",
        "Confidence": proba,
        "Arrondissement": f"{zone}_{gouv}",
        "Mois": mois,
        "Nbre d'intersection": intersections
    })

    if pred == 1:
        st.error("🔴 Zone Dangereuse")
    else:
        st.success("🟢 Zone Sûre")

    st.metric("Confiance", f"{proba*100:.2f}%")

# =========================
# CLEAR
# =========================
if st.button("🧹 Clear Predictions"):
    st.session_state.pred_points = []

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📉 Confusion Matrix")

if st.button("Afficher Matrix"):
    # Préparer les données comme lors de l'entraînement
    df_encoded, _ = prepare_features(df)
    
    X = df_encoded[feature_cols]
    y = df_encoded["Dangereux"]

    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    st.pyplot(fig)