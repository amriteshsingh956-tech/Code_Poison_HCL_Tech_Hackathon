"""Local ShopSmart spending prediction demo."""
import json

import pandas as pd
import streamlit as st

from scripts.predict import BUNDLE, MODEL, predict_batch, predict_customer
from scripts.spend_bands import interval_label
import joblib


def metadata(name):
    """Read metadata produced alongside the fitted pipeline."""
    return json.loads((MODEL.parent / name).read_text())


st.set_page_config(page_title="ShopSmart spend estimate", layout="centered")
st.title("Estimated customer spend")
try:
    schema = metadata("input_schema.json")
    metrics = metadata("metrics.json")
    config = metadata("configuration.json")
    if not BUNDLE.is_file():
        raise FileNotFoundError
except FileNotFoundError:
    st.error("Run the local training workflow in the README to create the model package.")
    st.stop()

model_name = {"Linear": "Linear Regression"}.get(config["model"], config["model"])
st.caption(f"{model_name} · Spending period and currency unconfirmed")
single, batch = st.tabs(["One customer", "Batch CSV"])
with single:
    with st.form("customer"):
        profile = {}
        example = metrics["reload_check"]["example"]
        for name in schema["required"]:
            label = name.replace("_", " ").capitalize()
            spec = schema["fields"][name]
            if isinstance(spec, list):
                profile[name] = st.selectbox(label, [""] + spec, index=spec.index(example[name])+1,
                                             format_func=lambda s: s or "Unknown")
            else:
                profile[name] = st.text_input(label + (" (years, inferred)" if "years" in spec else ""),
                                              value=str(example[name]))
        submit = st.form_submit_button("Estimate spend")
    if submit:
        try:
            prediction = predict_customer(profile)
            st.metric("Estimated spend", f"{prediction['Predicted_Amount_Spent']:,.2f}")
            band = prediction["Predicted_Amount_Spent_Range"]
            bundle = joblib.load(BUNDLE)
            st.metric("Spend range", band)
            st.caption(interval_label(band,bundle["cutoffs"]) + " spending units")
        except ValueError as error:
            st.error(str(error))
with batch:
    st.write("Required columns: " + ", ".join(schema["required"]))
    st.caption("Extra columns and identifiers stay in input order. Files are processed locally.")
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is not None:
        try:
            upload.seek(0)
            result = predict_batch(upload, dual=True)
            st.dataframe(result, hide_index=True)
            st.download_button("Download predictions", result.to_csv(index=False).encode("utf-8"),
                               "spend_predictions.csv", "text/csv")
        except (ValueError, UnicodeError, pd.errors.ParserError) as error:
            st.error(f"Check the CSV and upload it again. {error}")
st.subheader("Holdout performance")
st.caption(f"{metrics['test_rows']:,} reserved synthetic profiles")
for column, key, label in zip(st.columns(3), ["rmse","mae","r2"], ["RMSE","MAE","R²"]):
    column.metric(label, f"{metrics['linear'][key]:,.3f}" if key=="r2" else f"{metrics['linear'][key]:,.2f}")
st.caption("The model tended to underestimate spending among the highest-spending profiles.")
