import joblib
import pandas as pd

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load("model.pkl")
    print("✔ Model loaded successfully")
except Exception as e:
    print("❌ ERROR loading model:")
    print(e)
    exit()

# =========================
# CHECK TYPE
# =========================
print("\n📌 Model type:")
print(type(model))

# =========================
# CHECK IF FITTED
# =========================
try:
    print("\n📌 Model feature names:")
    print(model.feature_names_in_)
except Exception as e:
    print("⚠ No feature_names_in_ (maybe pipeline or old model)")

# =========================
# TEST FAKE INPUT
# =========================
try:
    test_data = pd.DataFrame([{
        "Zone": "test",
        "Gouvernorat": "test"
    }])

    pred = model.predict(test_data)
    print("\n✔ Prediction test OK:", pred)

except Exception as e:
    print("\n❌ Prediction FAILED:")
    print(e)