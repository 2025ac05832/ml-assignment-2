"""Classifier definitions and the shared preprocessing pipeline.

Every model is returned as a complete Pipeline (preprocessing + estimator) so
that cross-validation refits the preprocessing inside each fold. Fitting a scaler
on the whole dataset before splitting leaks test-fold statistics into training and
inflates the reported scores; building it this way makes that mistake impossible
rather than merely avoided by convention.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest (Ensemble)",
]


def build_preprocessor(numeric_cols: list[str],
                       categorical_cols: list[str]) -> ColumnTransformer:
    """Impute and scale numerics; impute and one-hot encode categoricals.

    RobustScaler is used rather than StandardScaler because it centres on the
    median and scales by the interquartile range. Several features in this
    dataset are strongly right-skewed with long upper tails, and a mean/standard
    deviation scaler lets those tails dominate the transform.
    """
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                       # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    branches = [(
        "numeric",
        Pipeline([("impute", SimpleImputer(strategy="median")),
                  ("scale", RobustScaler())]),
        numeric_cols,
    )]

    if categorical_cols:
        branches.append((
            "categorical",
            Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("encode", encoder)]),
            categorical_cols,
        ))

    return ColumnTransformer(branches)


def base_estimators() -> dict:
    """The five estimators the assignment requires, before preprocessing is attached.

    class_weight="balanced" is set where the estimator supports it. The dataset is
    roughly 63/37, and reweighting costs nothing on the majority class while
    materially improving recall on the minority one — which is the class that
    matters in a diagnostic setting.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=8, class_weight="balanced",
            random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def build_all(numeric_cols: list[str], categorical_cols: list[str]) -> dict:
    """Return {model name: Pipeline} for every required model."""
    estimators = base_estimators()
    return {
        name: Pipeline([
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("classifier", estimators[name]),
        ])
        for name in MODEL_ORDER
    }
