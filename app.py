import json
import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load():
    model = joblib.load("bulldozer_model.joblib")
    with open("bulldozer_prep.json") as f:
        prep= json.load(f)
    return model, prep
model, prep = load()
cats, medians, cols = prep["cats"], prep["medians"], prep["columns"]

def preprocess(df):
    df = df.copy()
    d = pd.to_datetime(df["saledate"])
    df["SaleYear"] = d.dt.year
    df["SaleMonth"] = d.dt.month
    df["SaleDay"] = d.dt.day
    df["SaleDayOfTheWeek"] = d.dt.dayofweek
    df["SaleDayOfTheYear"] = d.dt.dayofyear
    df = df.drop(columns = "saledate")
    for c, m in medians.items():
        df[c + "_is_missing"] = df[c].isnull()
        df[c]= df[c].fillna(m)
    for c, cat_list in cats.items():
        df[c + "_is_missing"] = df[c].isnull()
        df[c] = pd.Categorical(df[c], categories = cat_list).codes + 1
    return df
st.title("Bulldozer Price Predictor")
st.write("Upload a CSV of bulldozer sales (Kaggle Bluebook format) to get predicted sale prices.")

with open("sample_bulldozers.csv", "rb") as f:
    st.download_button("Download a sample CSV to try", f, "sample_bulldozers.csv")

file = st.file_uploader("Upload a CSV", type="csv")
if file:
    raw = pd.read_csv(file, low_memory=False).head(1000)
    if "saledate" not in raw.columns:
        st.error("The CSV needs a 'saledate' column.")
        st.stop()
    X = preprocess(raw)
    missing = [c for c in cols if c not in X.columns]
    if missing:
        st.error(f"Missing columns: {missing[:10]}")
        st.stop()
    preds = model.predict(X[cols])
    out = raw.copy()
    out["PredictedPrice"] = preds.round(0)
    st.success(f"Predicted prices for {len(out)} rows (max 1000)")
    if "SalePrice" in out.columns:
        mae = (out["SalePrice"] - out["PredictedPrice"]).abs().mean()
        st.write(f"Average error vs actual price: ${mae:,.0f}")
    st.dataframe(out)
    
