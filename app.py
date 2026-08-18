"""Streamlit front-end for ML Assignment 2.

Architectural note: this app holds no pickled models. It imports the same
`model` package that `run_experiment.py` uses and fits the pipelines at startup,
cached with `st.cache_resource` so the work happens once per session.

The trade is a couple of seconds on cold start in exchange for three things: the
repository carries no binary artifacts, the app can never drift out of sync with
the training code, and there is no possibility of a pickle written by one
scikit-learn version failing to load under another — which is the usual reason a
deployment dies on first boot.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model import classifiers, datasets, evaluation

st.set_page_config(page_title="Diagnostic Classifier Workbench",
                   page_icon="\U0001F52C", layout="wide")


# --------------------------------------------------------------------------
# Cached training
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Fitting models from source...")
def prepare(csv_path: str, target_col: str, holdout: float = 0.2):
    """Load the reference dataset and fit every pipeline on the training rows.

    The same stratified split and seed as run_experiment.py are reproduced here,
    so the rows sitting in test_data.csv are excluded from fitting. Fitting on the
    full frame instead would let the app score itself on data it had already seen
    and report a meaningless 1.0000.
    """
    ds = datasets.load(csv_path, target_col)
    encoder = LabelEncoder()
    y = encoder.fit_transform(ds.target)

    train_idx, _ = train_test_split(
        np.arange(len(ds.frame)), test_size=holdout,
        random_state=classifiers.RANDOM_STATE, stratify=y)

    pipelines = classifiers.build_all(ds.numeric_cols, ds.categorical_cols)
    for pipeline in pipelines.values():
        pipeline.fit(ds.features.iloc[train_idx], y[train_idx])

    return ds, encoder, pipelines


@st.cache_data(show_spinner="Cross-validating...")
def cross_validated(csv_path: str, target_col: str, folds: int, repeats: int):
    ds = datasets.load(csv_path, target_col)
    encoder = LabelEncoder()
    y = encoder.fit_transform(ds.target)
    pipelines = classifiers.build_all(ds.numeric_cols, ds.categorical_cols)
    return evaluation.evaluate_all(pipelines, ds.features, y, ds.is_binary,
                                   folds, repeats, classifiers.RANDOM_STATE)


def heatmap(matrix, class_names, title=""):
    fig, ax = plt.subplots(figsize=(1.1 * len(class_names) + 3.2,
                                    0.9 * len(class_names) + 2.8))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Controls")

try:
    ds, encoder, pipelines = prepare(datasets.DEFAULT_CSV, datasets.DEFAULT_TARGET)
except datasets.DatasetError as exc:
    st.error(f"Could not load the reference dataset: {exc}")
    st.stop()

CLASS_NAMES = [str(c) for c in encoder.classes_]
TARGET_COL = ds.target_col
FEATURE_COLS = list(ds.features.columns)

selected_model = st.sidebar.selectbox("Select a model", list(pipelines),
                                      key="model_choice")

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"],
    help=f"Needs the feature columns and a '{TARGET_COL}' column.")
use_bundled = st.sidebar.checkbox("Use the bundled test_data.csv", value=True)

st.sidebar.markdown("---")
folds = st.sidebar.slider("CV folds", 3, 10, 5)
repeats = st.sidebar.slider("CV repeats", 1, 5, 3)

st.sidebar.markdown("---")
summary_info = ds.summary()
st.sidebar.caption(
    f"Reference dataset  \n"
    f"{summary_info['instances']} rows, {summary_info['features']} features  \n"
    f"{'Binary' if ds.is_binary else 'Multi-class'}, "
    f"{summary_info['classes']} classes"
)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Diagnostic Classifier Workbench")
st.write(
    "Five classifiers fitted from source at startup — no pickled artifacts — and "
    "compared using repeated stratified cross-validation rather than a single "
    "hold-out split, so the reported spread shows which differences are real."
)

cv_tab, evaluate_tab, predictions_tab, data_tab = st.tabs(
    ["Cross-validated comparison", "Evaluate on a CSV", "Predictions", "Dataset"]
)

# resolve uploaded / bundled frame
frame = None
source_label = None
if uploaded is not None:
    frame = pd.read_csv(uploaded)
    source_label = uploaded.name
elif use_bundled:
    try:
        frame = pd.read_csv("test_data.csv")
        source_label = "test_data.csv (bundled)"
    except FileNotFoundError:
        frame = None

with cv_tab:
    st.subheader(f"Mean ± SD over {folds} folds × {repeats} repeats")
    st.caption(
        "Each cell is the average across every fold, with the standard deviation "
        "beside it. Two models whose intervals overlap are not meaningfully "
        "different on that metric, however the means are ordered."
    )
    summary, per_fold = cross_validated(datasets.DEFAULT_CSV,
                                        datasets.DEFAULT_TARGET, folds, repeats)
    st.dataframe(evaluation.format_summary(summary), width="stretch")

    metric_choice = st.selectbox("Show the spread for", evaluation.METRIC_NAMES,
                                 index=4, key="metric_choice")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.boxplot([per_fold[per_fold["model"] == m][metric_choice].to_numpy()
                for m in pipelines])
    ax.set_xticks(range(1, len(pipelines) + 1),
                  [m.split(" (")[0] for m in pipelines])
    ax.set_ylabel(metric_choice)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    st.pyplot(fig)

    best = summary[metric_choice].idxmax()
    st.success(f"Highest mean {metric_choice}: **{best}** "
               f"({summary.loc[best, metric_choice]:.4f} ± "
               f"{summary.loc[best, metric_choice + ' SD']:.4f})")

with evaluate_tab:
    if frame is None:
        st.info("Upload a CSV in the sidebar, or tick the bundled-data option.")
    elif TARGET_COL not in frame.columns:
        st.error(f"Column '{TARGET_COL}' not found. Found: {list(frame.columns)}")
    else:
        missing = [c for c in FEATURE_COLS if c not in frame.columns]
        if missing:
            st.error(f"Missing feature columns: {missing[:8]}"
                     f"{' ...' if len(missing) > 8 else ''}")
        else:
            st.caption(f"Source: {source_label} — {len(frame)} rows")
            lookup = {name: i for i, name in enumerate(CLASS_NAMES)}
            y_true = frame[TARGET_COL].astype(str).map(lookup)

            if y_true.isna().any():
                st.error(f"Some labels are outside the training classes "
                         f"{CLASS_NAMES}.")
            else:
                y_true = y_true.to_numpy()
                pipeline = pipelines[selected_model]
                X = frame[FEATURE_COLS]
                y_pred = pipeline.predict(X)
                y_proba = pipeline.predict_proba(X)

                scores = evaluation._score_fold(y_true, y_pred, y_proba,
                                                ds.is_binary)

                st.subheader(f"{selected_model} — evaluation metrics")
                for col, (label, value) in zip(st.columns(6), scores.items()):
                    col.metric(label,
                               "n/a" if np.isnan(value) else f"{value:.4f}")

                left, right = st.columns(2)
                with left:
                    st.subheader("Confusion matrix")
                    cm = confusion_matrix(y_true, y_pred,
                                          labels=list(range(len(CLASS_NAMES))))
                    st.pyplot(heatmap(cm, CLASS_NAMES))
                with right:
                    st.subheader("Classification report")
                    report = classification_report(
                        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                        target_names=CLASS_NAMES, output_dict=True,
                        zero_division=0)
                    st.dataframe(pd.DataFrame(report).transpose().round(4),
                                 width="stretch")

                st.session_state["preds"] = (frame, y_pred, y_proba)

with predictions_tab:
    if "preds" not in st.session_state:
        st.info("Run an evaluation first.")
    else:
        source_frame, y_pred, y_proba = st.session_state["preds"]
        out = source_frame.copy()
        out["predicted"] = [CLASS_NAMES[i] for i in y_pred]
        out["confidence"] = y_proba.max(axis=1).round(4)
        if TARGET_COL in out.columns:
            out["correct"] = out[TARGET_COL].astype(str) == out["predicted"]
            wrong = int((~out["correct"]).sum())
            st.metric("Misclassified rows", f"{wrong} of {len(out)}")
            if wrong and st.checkbox("Show only the mistakes"):
                out = out[~out["correct"]]
        st.dataframe(out.head(300), width="stretch")
        st.download_button("Download predictions as CSV",
                           out.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv")

with data_tab:
    st.subheader("Reference dataset")
    st.json(ds.summary())
    st.dataframe(ds.frame.head(50), width="stretch")

    st.subheader("Class separation by feature")
    feature = st.selectbox("Feature", FEATURE_COLS, key="feature_choice")
    if feature in ds.numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for cls in CLASS_NAMES:
            ax.hist(ds.frame[ds.frame[TARGET_COL].astype(str) == cls][feature],
                    bins=30, alpha=0.55, label=cls)
        ax.set_xlabel(feature)
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.dataframe(pd.crosstab(ds.frame[feature], ds.frame[TARGET_COL]),
                     width="stretch")
