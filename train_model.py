import pandas as pd
import joblib

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("dataset_final.xlsx")
df.columns = df.columns.str.strip()

# TARGET
df["Dangereux"] = (df["Tués"] > 0).astype(int)

# =========================
# FEATURES (SIMPLIFIÉES)
# =========================
FEATURES = [
    "Zone",
    "Gouvernorat",
    "Mois",
    "Sécurité",
    "Nbre d'intersection"
]

X = df[FEATURES]
y = df["Dangereux"]

# SAVE FEATURES
joblib.dump(FEATURES, "features.pkl")

# =========================
# PREPROCESS
# =========================
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

# =========================
# MODELS
# =========================
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

results = {}

# =========================
# TRAIN + SAVE
# =========================
for name, clf in models.items():

    model = Pipeline([
        ("prep", preprocess),
        ("clf", clf)
    ])

    score = cross_val_score(model, X, y, cv=5).mean()
    results[name] = score

    model.fit(X, y)
    joblib.dump(model, f"{name}.pkl")

print("✅ Training terminé")
print(results)