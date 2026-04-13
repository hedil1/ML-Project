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
    
    # GPS coordinates
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
    
    # Clean data
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").fillna(34.5)
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").fillna(9.5)
    
    df["Nbre d'intersection"] = pd.to_numeric(
        df.get("Nbre d'intersection", 0),
        errors="coerce"
    ).fillna(0)
    
    return df, coords

df, coords = load_data()

# =========================
# CLUSTERING
# =========================
@st.cache_resource
def create_clusters():
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_copy = df.copy()
    df_copy["Cluster"] = kmeans.fit_predict(df_copy[["Latitude", "Longitude"]])
    return df_copy, kmeans

df_clustered, kmeans = create_clusters()
df["Cluster"] = df_clustered["Cluster"]

# =========================
# LOAD MODEL
# =========================
model_name = st.sidebar.selectbox(
    "Choisir modèle",
    ["GradientBoosting", "XGBoost", "RandomForest", "SVM", "KNN"]
)

try:
    model = joblib.load(f"models/{model_name}.pkl")
    feature_cols = joblib.load("models/features.pkl")
    model_loaded = True
except FileNotFoundError:
    st.error(f"⚠️ Modèle {model_name} non trouvé. Veuillez d'abord entraîner les modèles.")
    model_loaded = False

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
    mapbox_style="light"
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
    mapbox_style="light",
    title="Densité des zones dangereuses"
)

st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# PREDICTION
# =========================
st.subheader("🔮 Nouvelle prédiction")

if model_loaded:
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
        
        # Add missing columns with default values
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
    if model_loaded:
        try:
            X = df[feature_cols]
            y = df["Dangereux"]
            y_pred = model.predict(X)
            
            cm = confusion_matrix(y, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                       xticklabels=["Safe", "Danger"],
                       yticklabels=["Safe", "Danger"])
            ax.set_xlabel("Prédiction")
            ax.set_ylabel("Réel")
            ax.set_title(f"Matrice de Confusion - {model_name}")
            
            st.pyplot(fig)
            
            # Show accuracy
            accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
            st.metric("Précision globale", f"{accuracy*100:.2f}%")
            
        except Exception as e:
            st.error(f"Erreur lors du calcul: {e}")
    else:
        st.warning("Modèle non disponible")

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Légende:**
    - 🔴 Rouge: Zone dangereuse
    - 🟢 Vert: Zone sûre
    
    **Modèle actif:** {}
    """.format(model_name if model_loaded else "Non chargé")
)