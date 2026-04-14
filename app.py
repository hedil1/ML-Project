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
# DICTIONNAIRE LIEU -> GOUVERNORAT
# =========================
lieu_gouvernorat = {
    "b.arkoub-b.b.regba": "Nabeul",
    "manouba-jedeida": "Manouba",
    "sidi salah-sfax": "Sfax",
    "bekalta-moknine": "Monastir",
    "entrée cheylus": "Nabeul",
    "el djem-hancha": "Mahdia",
    "ezzahra mahdia": "Mahdia",
    "k.sghira-msaken": "Sousse",
    "goulette-j.jelloud": "Tunis",
    "nassen-oudna": "Zaghouan",
    "b.arkoub.grombalia": "Nabeul",
    "tunis-manouba": "Tunis",
    "b.arada-laroussa": "Zaghouan",
    "ghardimaou": "Jendouba",
    "sortie tébourba": "Manouba",
    "k.sghira-enfidha": "Sousse",
    "arrêt s.messaoud": "Sfax",
    "skhira-aouinet": "Sfax",
    "k.sghira-sousse": "Sousse",
    "k.sghira-m.gare": "Sousse",
    "entrée gâafour": "Siliana",
    "grombalia-b.cedria": "Nabeul",
    "fahs-depienne": "Zaghouan",
    "pn mellaha": "Bizerte",
    "le kef-les salines": "Le Kef",
    "j.jelloud-b.kassa": "Tunis",
    "oudna-nassen": "Zaghouan",
    "nabeul-b.b.regba": "Nabeul",
    "j.jelloud-tunis": "Tunis",
    "khélidia": "Kairouan",
    "sortie bir el bey": "Tunis",
    "tindja-bizerte": "Bizerte",
    "s.abid-guergour": "Siliana",
    "kerker-msaken": "Sousse",
    "mahdia-mahdia z.t": "Mahdia",
    "entrée nassen": "Zaghouan",
    "entrée grombalia": "Nabeul",
    "pn de msaken": "Sousse",
    "les salines-sers": "Le Kef",
    "b.kassa-cheylus": "Tunis",
    "b.cedria-grombalia": "Nabeul",
    "depienne-fahs": "Zaghouan",
    "t.sfar-arrêt du stad": "Sfax",
    "b.salem-jendouba": "Jendouba",
    "monastir-s.sud": "Monastir",
    "b.ficha-b.b.regba": "Nabeul",
    "tejrouine-jérissa": "Le Kef",
    "s.ezzit-sfax": "Sfax",
    "sened-zannouch": "Gabès",
    "hamma-chatt": "Gabès",
    "gafsa-mg1": "Gafsa",
    "gafsa": "Gafsa",
    "pn bardo": "Tunis",
    "moknine-k.medyouni": "Monastir",
    "b.salem-s.smail": "Jendouba",
    "o.meliz-jendouba": "Jendouba",
    "s.sud-monastir": "Monastir",
    "entrée medjez": "Béja",
    "monastir-ksibet": "Monastir",
    "laroussa-gâafour": "Siliana",
    "fahs-b.arada": "Zaghouan",
    "sortie ghannouch": "Gabès",
    "j.jelloud-goulette": "Tunis",
    "ksar helal": "Monastir",
    "enfidha-m.gare": "Sousse",
    "fondouk jedid": "Nabeul",
    "fahs-bou arada": "Zaghouan",
    "m.gare-k.sghira": "Sousse",
    "h.lif-rades": "Ben Arous",
    "khéniss-ksibet": "Monastir",
    "menzel-k sghira": "Sousse",
    "omar khay-hammamet": "Nabeul",
    "b ficha-enfidha": "Sousse",
    "m.bourg-tindja": "Bizerte",
    "ksibet-moknine": "Monastir",
    "pn de bejaoua": "Jendouba",
    "pn de l'aéroport": "Tunis",
    "jend.ghardimaou": "Jendouba"
}

# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("dataset_final.xlsx")
    df.columns = df.columns.str.strip()
    
    df["Dangereux"] = (df["Tués"] > 0).astype(int)
    df["Arrondissement"] = df["Zone"] + "_" + df["Gouvernorat"]
    
    # Ajouter la colonne Lieu à partir du dictionnaire si elle n'existe pas
    if "Lieu" not in df.columns:
        # Essayer de mapper les zones existantes
        df["Lieu"] = df["Zone"]
    
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
    except Exception:
        return None

# =========================
# LOAD DATA
# =========================
df, coords = load_data()

# =========================
# SIDEBAR - MODEL SELECTION
# =========================
st.sidebar.header(" Configuration Machine Learning")

model_name = st.sidebar.selectbox(
    " Choisir le modèle de prédiction:",
    ["GradientBoosting", "XGBoost", "RandomForest", "SVM", "KNN"]
)

if st.sidebar.button("Ré-entraîner tous les modèles", type="primary"):
    if retrain_models():
        st.rerun()

model = load_model(model_name)

if model is None:
    st.info(" Modèle non disponible. Cliquez sur 'Ré-entraîner' pour créer les modèles.")

# =========================
# FEATURES
# =========================
try:
    feature_cols = joblib.load("models/features.pkl")
except:
    feature_cols = ["Zone", "Gouvernorat", "Mois", "Sécurité", "Nbre d'intersection"]


# =========================
# SÉLECTION PAR GOUVERNORAT (AFFICHE TOUS LES LIEUX)
# =========================
st.subheader(" Exploration par gouvernorat")

# Liste unique des gouvernorats
gouvernorats_list = sorted(df["Gouvernorat"].unique())

col_gouv1, col_gouv2 = st.columns([1, 2])

with col_gouv1:
    selected_gouv = st.selectbox(
        " Choisir un gouvernorat:",
        options=gouvernorats_list,
        help="Sélectionnez un gouvernorat pour voir tous ses lieux"
    )

# Trouver tous les lieux appartenant à ce gouvernorat
lieux_du_gouvernorat = [lieu for lieu, gouv in lieu_gouvernorat.items() if gouv == selected_gouv]

# Ajouter les zones du dataframe qui correspondent
df_lieux_gouv = df[df["Gouvernorat"] == selected_gouv]["Zone"].unique()
for lieu in df_lieux_gouv:
    if lieu not in lieux_du_gouvernorat:
        lieux_du_gouvernorat.append(lieu)

lieux_du_gouvernorat = sorted(set(lieux_du_gouvernorat))

with col_gouv2:
    if len(lieux_du_gouvernorat) > 0:
        selected_lieu = st.selectbox(
            f" Lieux dans {selected_gouv}:",
            options=lieux_du_gouvernorat,
            help=f"Liste des passages à niveau dans {selected_gouv}"
        )
    else:
        st.warning(f"Aucun lieu trouvé pour {selected_gouv}")
        selected_lieu = None

# =========================
# AFFICHAGE DES DÉTAILS DU LIEU SÉLECTIONNÉ
# =========================
if selected_lieu:
    # Filtrer les données pour ce lieu spécifique
    df_lieu = df[df["Zone"] == selected_lieu]
    
    if len(df_lieu) > 0:
        st.subheader(f" Détails pour: {selected_lieu}")
        
        # Statistiques du lieu
        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        with col_l1:
            st.metric("Total accidents", len(df_lieu))
        with col_l2:
            st.metric("Zones dangereuses", df_lieu["Dangereux"].sum())
        with col_l3:
            st.metric("Taux de danger", f"{df_lieu['Dangereux'].mean()*100:.1f}%")
        with col_l4:
            st.metric("Total tués", df_lieu["Tués"].sum())
        
        # Statut de dangerosité
        est_dangereuse = df_lieu["Dangereux"].sum() > 0
        if est_dangereuse:
            st.error(f" **{selected_lieu} est une ZONE DANGEREUSE**")
        else:
            st.success(f" **{selected_lieu} est une ZONE SÛRE**")
        
        # Détails par mois
        with st.expander(" Voir les détails par mois"):
            st.dataframe(df_lieu[["Mois", "Tués", "Blessés", "Dangereux", "Sécurité", "Nbre d'intersection"]], use_container_width=True)
        
        # Graphique d'évolution
        if len(df_lieu) > 1:
            fig_evol = px.line(df_lieu, x="Mois", y="Dangereux", 
                               title=f"Évolution du danger - {selected_lieu}",
                               markers=True)
            st.plotly_chart(fig_evol, use_container_width=True)
    else:
        st.warning(f" Aucune donnée disponible pour {selected_lieu}")
# =========================
# LISTE COMPLÈTE DES LIEUX PAR GOUVERNORAT
# =========================
with st.expander(" Voir tous les lieux par gouvernorat"):
    # Créer un DataFrame pour l'affichage
    lieux_data = []
    for lieu, gouv in sorted(lieu_gouvernorat.items()):
        # Vérifier si ce lieu existe dans les données
        df_check = df[df["Zone"] == lieu]
        if len(df_check) > 0:
            est_danger = df_check["Dangereux"].sum() > 0
            status = " Dangereuse" if est_danger else " Sûre"
        else:
            status = " Non trouvé"
        lieux_data.append({"Gouvernorat": gouv, "Lieu": lieu, "Statut": status})
    
    df_lieux = pd.DataFrame(lieux_data)
    st.dataframe(df_lieux, use_container_width=True)

# =========================
# HEATMAP INTERACTIVE PAR MOIS
# =========================
st.subheader(" Carte de chaleur des zones dangereuses")

selected_month = st.select_slider(
    " Choisissez un mois:",
    options=sorted(df["Mois"].unique()),
    value=sorted(df["Mois"].unique())[0]
)

df_month = df[df["Mois"] == selected_month].copy()

if len(df_month) > 0:
    fig_heat = px.density_mapbox(
        df_month,
        lat="Latitude",
        lon="Longitude",
        z="Dangereux",
        radius=20,
        center=dict(lat=34.5, lon=9.5),
        zoom=5.5,
        mapbox_style="open-street-map",
        title=f" Densité des zones dangereuses - {selected_month}",
        color_continuous_scale="Reds",
        opacity=0.7,
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
        st.subheader(f" Zones dangereuses en {selected_month}")
        st.dataframe(zones_danger, use_container_width=True)

# =========================
# PRÉDICTION PERSONNALISÉE
# =========================
st.subheader(" Prédiction personnalisée")

with st.expander(" Faire une prédiction", expanded=True):
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        zone_pred = st.selectbox("Zone/Lieu", sorted(lieu_gouvernorat.keys()), key="pred_zone")
        # Auto-compléter le gouvernorat
        gouv_pred_auto = lieu_gouvernorat.get(zone_pred, "Tunis")
        st.info(f"Gouvernorat: {gouv_pred_auto}")
        mois_pred = st.selectbox("Mois", df["Mois"].unique(), key="pred_mois")
    
    with col_p2:
        securite_pred = st.slider("Niveau de sécurité (0-10)", 0, 10, 5, key="pred_sec")
        intersections_pred = st.number_input("Nombre d'intersections", 0, 50, 5, key="pred_inter")
    
    if st.button(" Lancer la prédiction", type="primary", use_container_width=True):
        if model is None:
            st.warning(" Modèle non disponible. Veuillez d'abord ré-entraîner les modèles.")
        else:
            input_data = pd.DataFrame([{
                "Zone": zone_pred,
                "Gouvernorat": gouv_pred_auto,
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
                        st.error(f" **{zone_pred} est une ZONE DANGEREUSE** \n\nProbabilité: {proba:.2%}")
                    else:
                        st.success(f"**{zone_pred} est une ZONE SÛRE** \n\nProbabilité: {(1-proba)*100:.2f}%")
                
                with col_r2:
                    st.metric("Niveau de confiance", f"{proba*100:.2f}%")
                    st.info(f"📍 {zone_pred} - {gouv_pred_auto}")
                
            except Exception as e:
                st.error(f" Erreur de prédiction: {e}")

# =========================
# SIDEBAR INFO
# =========================
