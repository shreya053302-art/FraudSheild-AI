"""Model Center: model status and training workflow."""
from __future__ import annotations

import streamlit as st

from backend import fraud_detection as fd
from backend import preprocessing as pp
from backend.config import DEFAULT_DATASET, MODEL_PATH

from .styles import kpi_row, page_header


def _status(bundle) -> None:
    if not bundle:
        st.warning("No trained model found. Train one below. Until then FraudShield will not produce predictions.")
        return
    m = bundle["metrics"][bundle["selected"]]
    kpi_row([("Active model", bundle["selected"], f"trained {bundle['trained_at']}", "purple"),
             ("PR-AUC (test)", f"{m['pr_auc']:.3f}" if m["pr_auc"] is not None else "n/a", "main selection metric", "blue"),
             ("Recall / Precision", f"{m['recall']:.2f} / {m['precision']:.2f}", "held-out test set", "green")])
    st.caption(f"Dataset: {bundle['dataset_name']} · {bundle['n_rows']:,} rows · fraud rate {bundle['fraud_rate'] * 100:.3f}% "
               f"· features: {len(bundle['feature_cols'])} · decision threshold {bundle['threshold']:.3f}")
    if bundle["dataset_name"].lower().startswith("sample_synthetic"):
        st.warning("This model was trained on SYNTHETIC test data. Its metrics describe fake data, not real fraud detection.")


def render(bundle) -> None:
    page_header("Model Center", "Check model status or train a new model on your own dataset", "🧠")
    _status(bundle)
    st.divider()
    st.subheader("Train a model")
    st.markdown("Both **Random Forest** and **Logistic Regression** are trained with class weights for the "
                "imbalance, evaluated on a held-out test set, and the better one (by validation PR-AUC) is saved.")

    source = st.radio("Dataset source", ["data/creditcard.csv", "Upload a CSV"], horizontal=True, key="mc_source",
                      index=0 if DEFAULT_DATASET.exists() else 1)
    df, name = None, None
    if source == "data/creditcard.csv":
        if DEFAULT_DATASET.exists():
            if st.button("Load dataset", key="mc_load_local"):
                try:
                    st.session_state["mc_df"] = (pp.read_csv_safely(DEFAULT_DATASET, DEFAULT_DATASET.name), DEFAULT_DATASET.name)
                except ValueError as exc:
                    st.error(str(exc))
        else:
            st.info("data/creditcard.csv was not found. Download it from Kaggle (see README, section 'Get the dataset'), "
                    "put it in the data folder, or choose 'Upload a CSV' above.")
    else:
        up = st.file_uploader("Training CSV", type=["csv"], key="mc_upload")
        if up is not None:
            try:
                st.session_state["mc_df"] = (pp.read_csv_safely(up, up.name), up.name)
            except ValueError as exc:
                st.error(str(exc))

    if "mc_df" in st.session_state:
        df, name = st.session_state["mc_df"]
    if df is None:
        return

    target = pp.detect_target(df)
    binary_cols = [c for c in df.columns if pp.detect_target(df, c) == c]
    if not binary_cols:
        st.error("No 0/1 column found, so there is nothing to learn from. The dataset needs a target such as 'Class'.")
        return
    st.success(f"Loaded {name}: {len(df):,} rows × {df.shape[1]} columns.")
    c1, c2, c3 = st.columns(3)
    target = c1.selectbox("Target column (1 = fraud)", binary_cols, index=binary_cols.index(target) if target in binary_cols else 0,
                          key="mc_target")
    n_est = c2.slider("Random Forest trees", 50, 300, 150, 10, key="mc_trees")
    limit = c3.number_input("Max rows to use (0 = all)", min_value=0, value=0, step=10000, key="mc_rows",
                            help="Use 50,000 for a quick test run. Use 0 (all rows) for the final model.")
    fraud_n = int(df[target].sum())
    st.caption(f"Fraud rows: {fraud_n:,} ({fraud_n / len(df) * 100:.3f}%) · missing values in features: "
               f"{int(df.drop(columns=[target]).select_dtypes('number').isna().sum().sum()):,}")
    st.caption("Training on the full 284,807-row dataset can take a few minutes on a laptop.")

    if st.button("🚀 Train model", key="mc_train"):
        log = []
        with st.status("Training ...", expanded=True) as status:
            def say(msg):
                log.append(msg)
                st.write(msg)
            try:
                new = fd.train_models(df, target=target, n_estimators=n_est, max_rows=int(limit) or None,
                                      dataset_name=name, progress=say)
                status.update(label="Training complete", state="complete")
            except Exception as exc:
                status.update(label="Training failed", state="error")
                st.error(f"Training failed: {exc}")
                return
        st.cache_resource.clear()
        st.success(f"Saved {new['selected']} to {MODEL_PATH.name}. All pages now use it.")
        st.rerun()
