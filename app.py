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
# CHECK AND RETRAIN MODELS IF NEEDED
# =========================
@st.cache_resource
def get_models():
    """Charge ou ré-entraîne les modèles selon la compatibilité"""
    
    model_name = st.sidebar.selectbox(
        "Choisir modèle",
        ["GradientBoosting", "XGBoost", "RandomForest", "SVM", "KNN"]
    )
    
    # Vérifier si les modèles existent et sont compatibles
    try:
        # Tester le chargement d'un modèle
        test_model = joblib.load(f"models/{model_name}.pkl")
        st.success(f"✅ Modèle {model_name} chargé avec succès!")
        return test_model, model_name
        
    except Exception as e:
        if "scikit-learn" in str(e) or "version" in str(e):
            st.warning("⚠️ Modèles incompatibles détectés. Ré-entraînement automatique...")
            
            with st.spinner("🔄 Ré-entraînement des modèles en cours (cela peut prendre 2-3 minutes)..."):
                
                # Importer les modules nécessaires
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.pipeline import Pipeline
                from sklearn.compose import ColumnTransformer
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.svm import SVC
                from sklearn.neighbors import KNeighborsClassifier
                from xgboost import XGBClassifier
                
                # Charger les données
                df = pd.read_excel("dataset_final.xlsx")
                df.columns = df.columns.str.strip()
                df["Dangereux"] = (df["Tués"] > 0).astype(int)
                
                # Features
                FEATURES = ["Zone", "Gouvernorat", "Mois", "Sécurité", "Nbre d'intersection"]
                X = df[FEATURES]
                y = df["Dangereux"]
                
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
                st.rerun()
                
        else:
            st.error(f"❌ Erreur de chargement: {str(e)}")
            return None, None

# =========================
# LOAD DATA (CACHED)
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
    
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce").fillna(34.5)
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce").fillna(9.5)
    
    df["Nbre d'intersection"] = pd.to_numeric(
        df.get("Nbre d'intersection", 0),
        errors="coerce"
    ).fillna(0)
    
    return df, coords

# =========================
# MAIN APP
# =========================
st.set_page_config(page_title="🚦 Smart PN Dashboard", layout="wide")
st.title("🚦 Smart Dashboard - Zones Dangereuses 🇹🇳")

# Charger les données
df, coords = load_data()

# Charger ou entraîner les modèles
model, model_name = get_models()

# Si modèle non chargé, arrêter ici
if model is None:
    st.stop()

# Reste de votre code ici...
# (Votre code existant pour les stats, filtres, map, etc.)