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
st.sidebar.header(" Configuration")
model_name = st.sidebar.selectbox(
    "Choisir modèle",
    ["GradientBoosting", "XGBoost", "RandomForest", "SVM", "KNN"]
)




# =========================
# STATS
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("Total", len(df))
col2.metric("Danger", df["Dangereux"].sum())
col3.metric("Safe", len(df) - df["Dangereux"].sum())

# =========================
# FEATURES
# =========================
try:
    feature_cols = joblib.load("models/features.pkl")
except:
    feature_cols = ["Zone", "Gouvernorat", "Mois", "Sécurité", "Nbre d'intersection"]

# =========================
# HEATMAP INTERACTIVE PAR MOIS
# =========================
st.subheader("🔥 Analyse des zones dangereuses par mois")

# Sélecteur de mois
selected_month = st.select_slider(
    "📅 Choisissez un mois:",
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
        title=f"🔥 Densité des zones dangereuses - Mois: {selected_month}",
        color_continuous_scale="Reds",
        opacity=0.7,
        labels={"Dangereux": "Niveau de danger"}
    )
    
    fig_heat.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Statistiques du mois
    st.subheader(f"📊 Statistiques pour {selected_month}")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total accidents", len(df_month))
    with col_b:
        st.metric("Zones dangereuses", df_month["Dangereux"].sum())
    with col_c:
        st.metric("Taux de danger", f"{df_month['Dangereux'].mean()*100:.1f}%")
    
    # Liste des zones dangereuses du mois
    st.subheader(f"📍 Zones dangereuses en {selected_month}")
    zones_dangereuses = df_month[df_month["Dangereux"] == 1][["Zone", "Gouvernorat", "Tués", "Blessés"]]
    
    if len(zones_dangereuses) > 0:
        st.dataframe(zones_dangereuses, use_container_width=True)
        
        # Top 5 des plus dangereuses
        st.subheader("⚠️ Top 5 des zones les plus critiques")
        top_danger = df_month.nlargest(5, "Tués")[["Zone", "Gouvernorat", "Tués", "Blessés"]]
        st.dataframe(top_danger, use_container_width=True)
    else:
        st.info(f"✅ Aucune zone dangereuse enregistrée pour {selected_month}")
else:
    st.warning(f"⚠️ Aucune donnée pour le mois {selected_month}")

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
    securite = st.number_input("Sécurité (0-10)", 0, 10, 5)
    intersections = st.number_input("Nbre intersections", 0, 50, 5)

if st.button("🚀 Predict", type="primary"):
    # Créer le DataFrame d'entrée
    input_data = pd.DataFrame([{
        "Zone": zone,
        "Gouvernorat": gouv,
        "Mois": mois,
        "Sécurité": securite,
        "Nbre d'intersection": intersections
    }])
    
    # Afficher les données pour débogage
    with st.expander("📋 Détails de la prédiction"):
        st.write("**Données envoyées:**")
        st.dataframe(input_data)
        st.write(f"**Features attendues par le modèle:** {feature_cols}")
    
    try:
        # Prédiction
        pred = model.predict(input_data)[0]
        
        proba = 0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0].max()
        
        # Afficher le résultat
        st.markdown("---")
        if pred == 1:
            st.error(f"🔴 **ZONE DANGEREUSE** 🔴\n\nProbabilité: {proba:.2%}")
        else:
            st.success(f"🟢 **ZONE SÛRE** 🟢\n\nProbabilité: {(1-proba)*100:.2f}%")
        
        st.metric("Niveau de confiance", f"{proba*100:.2f}%")
        
        # Ajouter à la session (optionnel)
        lat, lon = coords.get(gouv, (34.5, 9.5))
        st.session_state.pred_points.append({
            "Latitude": lat,
            "Longitude": lon,
            "Prediction": "Danger" if pred == 1 else "Safe",
            "Confidence": proba,
            "Zone": zone,
            "Gouvernorat": gouv,
            "Mois": mois
        })
        
    except Exception as e:
        st.error(f"❌ Erreur de prédiction: {e}")
        st.write("Structure du modèle:", model)

# =========================
# CLEAR PREDICTIONS
# =========================
if st.button("🧹 Clear Predictions"):
    st.session_state.pred_points = []
    st.success("✅ Prédictions effacées!")
    st.rerun()

# =========================
# CONFUSION MATRIX
# =========================
with st.expander("📊 Voir la matrice de confusion"):
    if st.button("Afficher Matrice de Confusion", key="conf_matrix"):
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
    
    **Fonctionnalités:**
    - Heatmap interactive par mois
    - Liste des zones dangereuses
    - Prédiction personnalisée
    """
)