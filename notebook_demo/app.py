"""Interactive local interface for the best model recorded in Untitled5."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
import streamlit as st
from notebook_demo.model import OUT, FEATURES, predict, validate

st.set_page_config(page_title="ShopSmart | Customer spend", page_icon="🛍️", layout="wide")


@st.cache_resource
def load_bundle(modified):
    """Reload the local trusted model when its artifact changes."""
    return joblib.load(OUT / "model.pkl")


st.title("ShopSmart · Customer spend")
st.caption("Team Code Poison / HCLTech · Interactive notebook model demo")
if not (OUT / "model.pkl").exists():
    st.error("Model not found. Run the training command in notebook_demo/README.md first.")
    st.stop()
bundle = load_bundle((OUT / "model.pkl").stat().st_mtime_ns)
metrics, schema = bundle["metrics"], bundle["schema"]
with st.sidebar:
    st.header("Model overview")
    st.success("Linear Regression")
    st.write("Lowest recorded test RMSE among the models in Untitled5.ipynb.")
    st.write(f"Training: {metrics['train_rows']:,} rows  \nEvaluation: {metrics['test_rows']:,} rows")
    st.caption("Currency and spending period are undocumented. Estimates use dataset spending units.")
    st.warning("Synthetic-data prototype: purchase/visit timing is unconfirmed. This is not a verified future-spend forecast.")
    st.caption("Inputs stay local. Uploads are processed in memory.")

single, batch, evaluation = st.tabs(["Customer estimate", "Batch predictions", "Model evaluation"])
with single:
    st.subheader("Explore a customer profile")
    st.write("Adjust the eight inputs and select Estimate spend.")
    profile = {}
    with st.form("customer"):
        left, right = st.columns(2)
        with left:
            for name, spec in schema["numeric"].items():
                label = name.replace("_", " ") + (" (years)" if name == "Age" else "")
                profile[name] = st.number_input(label, min_value=0.0, value=spec["default"], step=1.0,
                                               help=f"Training range: {spec['min']:,.2f}–{spec['max']:,.2f}. Units follow the source CSV.", key=name)
        with right:
            for name, choices in schema["categorical"].items():
                profile[name] = st.selectbox(name.replace("_", " "), choices, key=name)
        submitted = st.form_submit_button("Estimate spend", type="primary")
    if submitted:
        prediction = predict(pd.DataFrame([profile]), bundle)["Predicted_Amount_Spent"].iloc[0]
        st.metric("Estimated customer spend", f"{prediction:,.2f}")
        st.caption(f"Held-out average absolute error: {metrics['test']['MAE']:,.2f} spending units. This is an aggregate error, not a confidence interval for this customer.")
        for name, spec in schema["numeric"].items():
            if not spec["min"] <= profile[name] <= spec["max"]:
                st.warning(f"{name} is outside the training range; this estimate extrapolates beyond observed data.")
        if prediction < 0:
            st.warning("The model returned a negative estimate. Treat this profile as unsupported; no clipping was applied.")
        with st.expander("Review input profile"):
            st.dataframe(pd.DataFrame([profile]), hide_index=True)

with batch:
    st.subheader("Estimate multiple customers")
    st.write("Upload a CSV with all eight feature columns. Extra columns and customer identifiers are preserved.")
    sample = {**{k: v["default"] for k, v in schema["numeric"].items()},
              **{k: v[0] for k, v in schema["categorical"].items()}}
    st.download_button("Download input template", pd.DataFrame([sample])[FEATURES].to_csv(index=False), "shopsmart_input_template.csv", "text/csv")
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    st.caption("Blank values use training-fitted imputation. Unseen categories use the encoder's baseline and may reduce reliability.")
    if uploaded:
        try:
            frame = pd.read_csv(io.BytesIO(uploaded.getvalue()), dtype=str)
            checked = validate(frame)
            for name, choices in schema["categorical"].items():
                if (~checked[name].isin(choices) & checked[name].notna()).any():
                    st.warning(f"Unseen values in {name}: encoded using the baseline. Check spelling and applicability.")
            result = predict(frame, bundle)
            st.success(f"Predicted {len(result):,} customer rows.")
            st.dataframe(result, hide_index=True)
            st.download_button("Download predictions", result.to_csv(index=False), "shopsmart_predictions.csv", "text/csv")
        except (ValueError, UnicodeError, pd.errors.ParserError) as error:
            st.error(str(error))

with evaluation:
    st.subheader("Evaluation on the notebook's held-out split")
    for column, (name, value) in zip(st.columns(4), metrics["test"].items()):
        column.metric(name, f"{value:,.4f}" if name == "R2" else f"{value:,.2f}")
    st.write(f"MAE means predictions differed from actual spending by {metrics['test']['MAE']:,.2f} units on average.")
    st.caption("RMSE penalizes large errors; lower is better. R² measures explained variation, not classification accuracy. MSE uses squared spending units.")
    st.dataframe(pd.DataFrame([metrics["test"], metrics["benchmark"]], index=["Linear Regression", "Predict training mean"]), width="stretch")
    st.image(str(OUT / "evaluation.png"), caption="Local reproduction · reserved 20% split · residual = actual − predicted")
    st.write(f"Training R²: {metrics['train_r2']:.4f} · Split seed: {metrics['seed']}")
    st.info("The notebook compared models on this test split, so these are retrospective comparison scores, not an independent post-selection estimate. New unseen data is needed for that assessment.")
    with st.expander("Notebook comparison — recorded outputs, not rerun searches"):
        st.dataframe(pd.DataFrame({"Model": ["Linear Regression", "Tuned Random Forest", "Tuned Decision Tree", "Decision Tree"],
                                   "Notebook RMSE": [844.203549701174, 981.2599458804942, 1237.5218404113104, 1442.0010458471206]}), hide_index=True)
        st.caption("Source: Untitled5.ipynb saved outputs. Only the winning Linear Regression is refitted for this demo.")
    st.download_button("Download evaluation metrics", (OUT / "metrics.json").read_bytes(), "evaluation_metrics.json", "application/json")
