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
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False

# =========================
# LOAD DATA (CACHED)
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

# =========================
# RETRAIN MODELS FUNCTION
# =========================
def retrain_models():
    """Fonction pour ré-entraîner les modèles"""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    
    with st.spinner("🔄 Ré-entraînement des modèles en cours... (2-3 minutes)"):
        # Charger les données
        df = pd.read_excel("dataset_final.xlsx")
        df.columns = df.columns.str.strip()
        df["Dangereux"] = (df["Tués"] > 0).astype(int)
        
        # Features
        FEATURES = ["Zone", "Gouvernorat", "Mois", "Sécurité", "Nbre d'intersection"]
        X = df[FEATURES]
        y = df["Dangereux"]
        
        # Créer le dossier models
        os.makedirs("models", exist_ok=True)
        
        # Sauvegarder les features
        joblib.dump(FEATURES, "models/features.pkl")
        
        # Prétraitement
        cat_cols = X.select_dtypes(include="object").columns
        num_cols = X.select_dtypes(exclude="object").columns
        
        preprocess = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ])
        
        # Modèles
        models_dict = {
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }
        
        # Entraînement
        progress_bar = st.progress(0)
        for idx, (name, clf) in enumerate(models_dict.items()):
            st.write(f"📈 Entraînement de {name}...")
            model = Pipeline([("prep", preprocess), ("clf", clf)])
            model.fit(X, y)
            joblib.dump(model, f"models/{name}.pkl")
            progress_bar.progress((idx + 1) / len(models_dict))
        
        st.success("✅ Tous les modèles ont été ré-entraînés avec succès!")
        return True

# =========================
# LOAD MODEL
# =========================
def load_model(model_name):
    """Charge un modèle spécifique"""
    try:
        model = joblib.load(f"models/{model_name}.pkl")
        return model, None
    except Exception as e:
        return None, str(e)

# =========================
# LOAD DATA
# =========================
df, coords = load_data()

# =========================
# SIDEBAR - MODEL SELECTION
# =========================
st.sidebar.header("🤖 Configuration")
model_name = st.sidebar.selectbox(
    "Choisir modèle",
    ["GradientBoosting", "XGBoost", "RandomForest", "SVM", "KNN"]
)

# =========================
# CHECK AND LOAD MODEL
# =========================
model, error = load_model(model_name)

if error:
    st.warning(f"⚠️ Modèle non compatible ou manquant: {error}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Ré-entraîner tous les modèles", type="primary"):
            if retrain_models():
                st.rerun()
    with col2:
        st.info("💡 Les modèles seront entraînés avec la version actuelle de scikit-learn")
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
st.sidebar.header("🎯 Filtres")

arr_options = df["Arrondissement"].unique()
arr = st.sidebar.selectbox("Arrondissement", arr_options)

mois_options = df["Mois"].unique()
mois_filter = st.sidebar.selectbox("Mois", mois_options)

df_filtered = df[
    (df["Arrondissement"] == arr) &
    (df["Mois"] == mois_filter)
].copy()

# =========================
# MAP
# =========================
st.subheader("🗺️ Carte intelligente")

map_df = df_filtered.copy()

# Add predictions
if len(st.session_state.pred_points) > 0:
    pred_df = pd.DataFrame(st.session_state.pred_points)
    map_df = pd.concat([map_df, pred_df], ignore_index=True)

# Set colors
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
    hover_data={"Latitude": False, "Longitude": False, "Color": False},
    zoom=6,
    height=550,
    color_continuous_scale=["green", "red"],
    title="📍 Zones Dangereuses"
)

fig_map.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),
    mapbox_style="open-street-map"
)

st.plotly_chart(fig_map, use_container_width=True)

# =========================
# HEATMAP
# =========================
st.subheader("🔥 Heatmap des accidents")

fig_heat = px.density_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    z="Dangereux",
    radius=15,
    center=dict(lat=34.5, lon=9.5),
    zoom=5,
    mapbox_style="open-street-map",
    title="Densité des zones dangereuses"
)

st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# FEATURES
# =========================
try:
    feature_cols = joblib.load("models/features.pkl")
except:
    feature_cols = ["Zone", "Gouvernorat", "Mois", "Sécurité", "Nbre d'intersection"]

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

if st.button("🚀 Predict", type="primary"):
    # Create input DataFrame
    input_dict = {
        "Zone": zone,
        "Gouvernorat": gouv,
        "Mois": mois,
        "Sécurité": securite,
        "Nbre d'intersection": intersections
    }
    
    # Add missing columns
    for col in feature_cols:
        if col not in input_dict:
            input_dict[col] = 0
    
    input_data = pd.DataFrame([input_dict])
    input_data = input_data[feature_cols]
    
    # Predict
    pred = model.predict(input_data)[0]
    
    proba = 0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0].max()
    
    # Get coordinates
    lat, lon = coords.get(gouv, (34.5, 9.5))
    lat += np.random.uniform(-0.03, 0.03)
    lon += np.random.uniform(-0.03, 0.03)
    
    # Store prediction
    st.session_state.pred_points.append({
        "Latitude": lat,
        "Longitude": lon,
        "Prediction": "Danger" if pred == 1 else "Safe",
        "Confidence": proba,
        "Arrondissement": f"{zone}_{gouv}",
        "Mois": mois,
        "Nbre d'intersection": intersections,
        "Dangereux": pred
    })
    
    # Show result
    if pred == 1:
        st.error("🔴 **Zone Dangereuse** - Soyez vigilant!")
    else:
        st.success("🟢 **Zone Sûre** - Pas de danger immédiat")
    
    st.metric("Confiance", f"{proba*100:.2f}%")
    
    st.rerun()

# =========================
# CLEAR PREDICTIONS
# =========================
if st.button("🧹 Clear Predictions"):
    st.session_state.pred_points = []
    st.rerun()

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📊 Performance du modèle")

if st.button("Afficher Matrice de Confusion"):
    try:
        df_temp = pd.read_excel("dataset_final.xlsx")
        df_temp.columns = df_temp.columns.str.strip()
        df_temp["Dangereux"] = (df_temp["Tués"] > 0).astype(int)
        
        X_all = df_temp[feature_cols]
        y_all = df_temp["Dangereux"]
        
        y_pred = model.predict(X_all)
        
        cm = confusion_matrix(y_all, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                   xticklabels=["Safe", "Danger"],
                   yticklabels=["Safe", "Danger"])
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réel")
        ax.set_title(f"Matrice de Confusion - {model_name}")
        
        st.pyplot(fig)
        
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        st.metric("Précision globale", f"{accuracy*100:.2f}%")
        
    except Exception as e:
        st.error(f"Erreur lors du calcul: {e}")

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Légende:**
    - 🔴 Rouge: Zone dangereuse
    - 🟢 Vert: Zone sûre
    
    **Modèle actif:** {model_name}
    """
)