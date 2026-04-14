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
st.set_page_config(page_title=" Smart PN Dashboard", layout="wide")
st.title(" Smart Dashboard - Zones Dangereuses 🇹🇳")

# =========================
# SESSION STATE
# =========================
if "pred_points" not in st.session_state:
    st.session_state.pred_points = []
if "selected_zone" not in st.session_state:
    st.session_state.selected_zone = None
if "selected_gouv" not in st.session_state:
    st.session_state.selected_gouv = None

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
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    
    with st.spinner(" Ré-entraînement des modèles en cours... (2-3 minutes)"):
        df = pd.read_excel("dataset_final.xlsx")
        df.columns = df.columns.str.strip()
        df["Dangereux"] = (df["Tués"] > 0).astype(int)
        
        FEATURES = ["Zone", "Gouvernorat", "Mois", "Sécurité", "Nbre d'intersection"]
        X = df[FEATURES]
        y = df["Dangereux"]
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(FEATURES, "models/features.pkl")
        
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
        
        models_dict = {
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }
        
        progress_bar = st.progress(0)
        for idx, (name, clf) in enumerate(models_dict.items()):
            st.write(f" Entraînement de {name}...")
            model = Pipeline([("prep", preprocess), ("clf", clf)])
            model.fit(X, y)
            joblib.dump(model, f"models/{name}.pkl")
            progress_bar.progress((idx + 1) / len(models_dict))
        
        st.success(" Tous les modèles ont été ré-entraînés avec succès!")
        return True

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model(model_name):
    """Charge un modèle spécifique"""
    try:
        model = joblib.load(f"models/{model_name}.pkl")
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

# =========================
# LOAD DATA
# =========================
df, coords = load_data()

# =========================
# SIDEBAR - MODEL SELECTION
# =========================
st.sidebar.header(" Configuration Machine Learning")

# Sélection du modèle d'entraînement
model_name = st.sidebar.selectbox(
    "🎯 Choisir le modèle de prédiction:",
    ["GradientBoosting", "XGBoost", "RandomForest", "SVM", "KNN"],
    help="Sélectionnez l'algorithme ML pour les prédictions"
)

# Bouton pour ré-entraîner
if st.sidebar.button(" Ré-entraîner tous les modèles", type="primary"):
    if retrain_models():
        st.rerun()

# Charger le modèle
model = load_model(model_name)

if model is None:
    st.warning(" Modèle non disponible. Cliquez sur 'Ré-entraîner' pour créer les modèles.")
    st.stop()

# =========================
# FEATURES
# =========================
try:
    feature_cols = joblib.load("models/features.pkl")
except:
    feature_cols = ["Zone", "Gouvernorat", "Mois", "Sécurité", "Nbre d'intersection"]

# =========================
# STATS GÉNÉRALES
# =========================
st.subheader(" Statistiques générales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total accidents", len(df))
col2.metric("Zones dangereuses", df["Dangereux"].sum())
col3.metric("Zones sûres", len(df) - df["Dangereux"].sum())
col4.metric("Taux de danger", f"{df['Dangereux'].mean()*100:.1f}%")

# =========================
# SÉLECTION DE ZONE INTERACTIVE
# =========================
st.subheader(" Exploration par zone")

col_zone1, col_zone2 = st.columns(2)

with col_zone1:
    # Sélection de la zone
    zone_selected = st.selectbox(
        " Choisir une zone:",
        options=sorted(df["Zone"].unique()),
        index=0 if st.session_state.selected_zone is None else list(df["Zone"].unique()).index(st.session_state.selected_zone) if st.session_state.selected_zone in df["Zone"].unique() else 0
    )
    st.session_state.selected_zone = zone_selected

# Filtrer les gouvernorats de la zone sélectionnée
gouv_options = df[df["Zone"] == zone_selected]["Gouvernorat"].unique()
with col_zone2:
    gouv_selected = st.selectbox(
        " Choisir un gouvernorat:",
        options=sorted(gouv_options),
        index=0 if st.session_state.selected_gouv is None else list(gouv_options).index(st.session_state.selected_gouv) if st.session_state.selected_gouv in gouv_options else 0
    )
    st.session_state.selected_gouv = gouv_selected

# Afficher les informations de la zone sélectionnée
df_zone = df[(df["Zone"] == zone_selected) & (df["Gouvernorat"] == gouv_selected)]

if len(df_zone) > 0:
    st.subheader(f" Informations pour {zone_selected} - {gouv_selected}")
    
    # Statistiques de la zone
    col_z1, col_z2, col_z3, col_z4 = st.columns(4)
    with col_z1:
        st.metric("Total accidents", len(df_zone))
    with col_z2:
        st.metric("Zones dangereuses", df_zone["Dangereux"].sum())
    with col_z3:
        st.metric("Taux de danger", f"{df_zone['Dangereux'].mean()*100:.1f}%")
    with col_z4:
        st.metric("Total tués", df_zone["Tués"].sum())
    
    # Statut de dangerosité
    est_dangereuse = df_zone["Dangereux"].sum() > 0
    if est_dangereuse:
        st.error(f" **{zone_selected} - {gouv_selected} est une ZONE DANGEREUSE**")
    else:
        st.success(f" **{zone_selected} - {gouv_selected} est une ZONE SÛRE**")
    
    # Afficher les données détaillées
    with st.expander(" Voir les détails des accidents"):
        st.dataframe(df_zone[["Mois", "Tués", "Blessés", "Dangereux", "Sécurité", "Nbre d'intersection"]], use_container_width=True)

# =========================
# HEATMAP INTERACTIVE PAR MOIS
# =========================
st.subheader(" Carte de chaleur des zones dangereuses")

# Sélecteur de mois avec slider
selected_month = st.select_slider(
    " Choisissez un mois:",
    options=sorted(df["Mois"].unique()),
    value=sorted(df["Mois"].unique())[0]
)

# Filtrer les données
df_month = df[df["Mois"] == selected_month].copy()

if len(df_month) > 0:
    # Heatmap
    fig_heat = px.density_mapbox(
        df_month,
        lat="Latitude",
        lon="Longitude",
        z="Dangereux",
        radius=20,
        center=dict(lat=34.5, lon=9.5),
        zoom=5.5,
        mapbox_style="open-street-map",
        title=f"🔥 Densité des zones dangereuses - {selected_month}",
        color_continuous_scale="Reds",
        opacity=0.7,
        labels={"Dangereux": "Niveau de danger"},
        hover_data={"Zone": True, "Gouvernorat": True, "Tués": True}
    )
    
    fig_heat.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Statistiques du mois
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total accidents", len(df_month))
    with col_m2:
        st.metric("Zones dangereuses", df_month["Dangereux"].sum())
    with col_m3:
        st.metric("Taux de danger", f"{df_month['Dangereux'].mean()*100:.1f}%")
    
    # Liste des zones dangereuses du mois
    zones_danger = df_month[df_month["Dangereux"] == 1][["Zone", "Gouvernorat", "Tués", "Blessés", "Sécurité"]]
    
    if len(zones_danger) > 0:
        st.subheader(f"⚠️ Zones dangereuses en {selected_month}")
        st.dataframe(zones_danger, use_container_width=True)
        
        # Top 5 des plus dangereuses
        top_danger = df_month.nlargest(5, "Tués")[["Zone", "Gouvernorat", "Tués", "Blessés"]]
        st.subheader("🔥 Top 5 des zones les plus critiques")
        st.dataframe(top_danger, use_container_width=True)
    else:
        st.success(f" Aucune zone dangereuse enregistrée pour {selected_month}")

# =========================
# PRÉDICTION PERSONNALISÉE
# =========================
st.subheader("🔮 Prédiction personnalisée")

with st.expander(" Faire une prédiction pour une nouvelle zone", expanded=True):
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        zone_pred = st.selectbox("Zone", df["Zone"].unique(), key="pred_zone")
        gouv_pred = st.selectbox("Gouvernorat", df["Gouvernorat"].unique(), key="pred_gouv")
        mois_pred = st.selectbox("Mois", df["Mois"].unique(), key="pred_mois")
    
    with col_p2:
        securite_pred = st.slider("Niveau de sécurité (0-10)", 0, 10, 5, key="pred_sec")
        intersections_pred = st.number_input("Nombre d'intersections", 0, 50, 5, key="pred_inter")
    
    if st.button(" Lancer la prédiction", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "Zone": zone_pred,
            "Gouvernorat": gouv_pred,
            "Mois": mois_pred,
            "Sécurité": securite_pred,
            "Nbre d'intersection": intersections_pred
        }])
        
        try:
            pred = model.predict(input_data)[0]
            
            proba = 0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0].max()
            
            st.markdown("---")
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                if pred == 1:
                    st.error(f" **ZONE DANGEREUSE** \n\nProbabilité: {proba:.2%}")
                else:
                    st.success(f" **ZONE SÛRE** \n\nProbabilité: {(1-proba)*100:.2f}%")
            
            with col_r2:
                st.metric("Niveau de confiance", f"{proba*100:.2f}%")
                st.info(f" {zone_pred} - {gouv_pred} | Mois: {mois_pred}")
            
        except Exception as e:
            st.error(f" Erreur de prédiction: {e}")

# =========================
# PERFORMANCE DU MODÈLE
# =========================
with st.expander(" Performance du modèle actuel"):
    if st.button("Afficher la matrice de confusion", key="show_cm"):
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
            st.error(f"Erreur: {e}")

