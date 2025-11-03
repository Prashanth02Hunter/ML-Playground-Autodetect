import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib, sklearn

# ------------------------------
# Version-robust OneHotEncoder
# ------------------------------
major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
ohe_kwargs = dict(handle_unknown="ignore")
if (major, minor) >= (1, 2):
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False

# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="ML Playground (Clean)", page_icon="🎛️", layout="wide")
st.title("🎛️ ML Playground — Auto-Detect, Clean & Future-Proof")

with st.expander("About", expanded=False):
    st.markdown(
        "- Upload a CSV and choose the **target** column.\n"
        "- The app **auto-detects** if it's a classification or regression problem and locks model choices.\n"
        "- Fully version-proof: OneHotEncoder & RMSE work on any scikit-learn.\n"
        "- Streamlit-2025 ready: no deprecated `use_container_width`; Plotly uses `config={\"responsive\": True}`.\n"
    )

# ------------------------------
# Data Loading
# ------------------------------
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sample = st.sidebar.selectbox("...or use a sample", ["None", "data/sample_classification.csv", "data/sample_regression.csv"])

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif sample != "None":
    try:
        df = load_data(sample)
    except Exception as e:
        st.error(f"Failed to load sample: {e}")
        st.stop()
else:
    st.info("Upload a CSV or choose a sample to start.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(), width='stretch')
st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

# ------------------------------
# Target & Task Inference
# ------------------------------
st.sidebar.header("2) Target")
target = st.sidebar.selectbox("Select target column", options=df.columns.tolist())

def infer_task(series: pd.Series) -> str:
    # numeric but few uniques -> classification; many uniques -> regression
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "classification" if series.nunique() <= max(10, int(0.02 * len(series))) else "regression"
    if pd.api.types.is_float_dtype(series):
        return "regression" if series.nunique() > 20 else "classification"
    # strings/categories -> classification
    return "classification"

if target is None:
    st.warning("Pick a target column to continue.")
    st.stop()

y_series = df[target]
task = infer_task(y_series)
st.info(f"**Detected task:** `{task}` based on target `{target}`.")

# Features
feature_cols = [c for c in df.columns if c != target]
X = df[feature_cols].copy()
y = y_series.copy()

# DType split
num_cols = X.select_dtypes(include=["int64","float64","int32","float32","int","float"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ------------------------------
# Preprocessing
# ------------------------------
numeric_tf = Pipeline(steps=[("scaler", StandardScaler())])
categorical_tf = Pipeline(steps=[("ohe", OneHotEncoder(**ohe_kwargs))])
preprocess = ColumnTransformer(
    transformers=[("num", numeric_tf, num_cols), ("cat", categorical_tf, cat_cols)],
    remainder="drop"
)

# ------------------------------
# Model Choice (Locked by Task)
# ------------------------------
st.sidebar.header("3) Model & Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

if task == "classification":
    model_options = ["LogisticRegression", "RandomForestClassifier"] + (["XGBClassifier"] if HAS_XGB else [])
else:
    model_options = ["LinearRegression", "RandomForestRegressor"] + (["XGBRegressor"] if HAS_XGB else [])

model_name = st.sidebar.selectbox("Model", model_options)

with st.sidebar.expander("Hyperparameters", expanded=False):
    if model_name in ["RandomForestClassifier", "RandomForestRegressor"]:
        n_estimators = st.number_input("n_estimators", 50, 1000, 200, 50)
        max_depth = st.number_input("max_depth (0=none)", 0, 50, 0, 1)
    elif model_name == "LogisticRegression":
        C = st.number_input("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01)
        max_iter = st.number_input("max_iter", 100, 5000, 1000, 100)
    elif model_name == "LinearRegression":
        st.caption("LinearRegression has no major hyperparameters.")
    elif model_name in ["XGBClassifier", "XGBRegressor"] and HAS_XGB:
        xgb_lr = st.number_input("learning_rate", 0.01, 1.0, 0.1, 0.01)
        xgb_n = st.number_input("n_estimators", 50, 2000, 300, 50)
        xgb_md = st.number_input("max_depth", 1, 20, 6, 1)

def make_model():
    if task == "classification":
        if model_name == "LogisticRegression":
            return LogisticRegression(C=C, max_iter=max_iter)
        if model_name == "RandomForestClassifier":
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=(None if max_depth==0 else max_depth), random_state=random_state)
        if model_name == "XGBClassifier" and HAS_XGB:
            return XGBClassifier(learning_rate=xgb_lr, n_estimators=xgb_n, max_depth=xgb_md, subsample=0.9, colsample_bytree=0.9, random_state=random_state, eval_metric="logloss")
    else:
        if model_name == "LinearRegression":
            return LinearRegression()
        if model_name == "RandomForestRegressor":
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth==0 else max_depth), random_state=random_state)
        if model_name == "XGBRegressor" and HAS_XGB:
            return XGBRegressor(learning_rate=xgb_lr, n_estimators=xgb_n, max_depth=xgb_md, subsample=0.9, colsample_bytree=0.9, random_state=random_state)
    raise ValueError("Unsupported model or missing dependency.")

pipe = Pipeline(steps=[("preprocess", preprocess), ("model", make_model())])

# ------------------------------
# Train
# ------------------------------
train_btn = st.button("🚀 Train model")
if train_btn:
    try:
        strat = y if task == "classification" and y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        st.success("Training complete!")

        if task == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            c1, c2 = st.columns(2)
            with c1: st.metric("Accuracy", f"{acc:.3f}")
            with c2: st.metric("F1 (weighted)", f"{f1:.3f}")

            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual", color="Count"))
            st.subheader("Confusion Matrix")
            st.plotly_chart(fig_cm, config={"responsive": True})

            # ROC Curve (binary)
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(pipe.named_steps["model"], "predict_proba"):
                        y_prob = pipe.predict_proba(X_test)[:, 1]
                    else:
                        y_prob = None
                    if y_prob is not None:
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = roc_auc_score(y_test, y_prob)
                        fig = px.area(x=fpr, y=tpr,
                                      labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                      title=f'ROC Curve (AUC={roc_auc:.3f})')
                        st.subheader("ROC Curve (binary)")
                        st.plotly_chart(fig, config={"responsive": True})
                except Exception as e:
                    st.info(f"ROC not available: {e}")

        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            try:
                rmse = mean_squared_error(y_test, y_pred, squared=False)
            except TypeError:
                rmse = mean_squared_error(y_test, y_pred) ** 0.5
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("R²", f"{r2:.3f}")
            with c2: st.metric("MAE", f"{mae:.3f}")
            with c3: st.metric("RMSE", f"{rmse:.3f}")

            # Residual plot
            residuals = y_test - y_pred
            fig_res = px.scatter(x=y_pred, y=residuals,
                                 labels={"x": "Predicted", "y": "Residuals"},
                                 title="Residual Plot")
            st.subheader("Residuals")
            st.plotly_chart(fig_res, config={"responsive": True})

        # Download artifacts
        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        st.download_button("💾 Download trained model (.joblib)", data=buf.getvalue(), file_name="model_pipeline.joblib")

        pred_df = X_test.copy()
        pred_df[target] = y_test
        pred_df["prediction"] = y_pred
        st.download_button("📥 Download predictions CSV", data=pred_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Training failed: {e}")
