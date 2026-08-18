# Breast Cancer Diagnosis — ML Assignment 2

**Name:** Shree Krishna Mishra
**BITS ID:** 2025ac05832
**Programme:** M.Tech (AIML) — Work Integrated Learning Programmes
**Course:** Machine Learning

Five classification models evaluated with repeated stratified cross-validation on
the Wisconsin Diagnostic Breast Cancer dataset, with a Streamlit front-end that
fits the models from source rather than loading pickled artifacts.

---

## a. Problem Statement

Given thirty numeric measurements computed from a digitised image of a fine
needle aspirate of a breast mass — radius, texture, perimeter, area, smoothness
and related shape descriptors, each summarised as a mean, a standard error and a
worst-case value — classify the mass as **malignant** or **benign**. This is a
binary classification problem.

The costs are asymmetric in a way that shapes the whole evaluation. Missing a
malignant case is far worse than flagging a benign one for further review, so
recall on the malignant class matters more than headline accuracy. Every model
here is configured with `class_weight="balanced"` where the estimator supports
it, for that reason.

---

## b. Dataset Description

| Property        | Value                                                                     |
| --------------- | ------------------------------------------------------------------------- |
| Source          | Wisconsin Diagnostic Breast Cancer (WDBC), UCI ML Repository              |
| URL             | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Accessed via    | `sklearn.datasets.load_breast_cancer` — bundled, no download              |
| Instances       | 569                                                                       |
| Features        | 30 (meets the ≥12 requirement)                                            |
| Target variable | `diagnosis`                                                               |
| Classes         | `benign` (357), `malignant` (212)                                         |
| Class balance   | 63 / 37 — mildly imbalanced                                               |
| Missing values  | None                                                                      |
| Feature types   | 30 numeric, 0 categorical                                                 |

### Feature structure

The thirty features are ten base measurements, each reported three ways:

| Base measurement                                                                                                  | Reported as                                                                        |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension | `mean_*`, `*_error` (standard error), `worst_*` (mean of the three largest values) |

This structure matters for interpretation: `mean_radius`, `radius_error` and
`worst_radius` are three views of one underlying quantity and are strongly
correlated with each other. Any observation about feature independence has to
reckon with that.

### Preprocessing

- **`RobustScaler` rather than `StandardScaler`.** Several features — `area_error`
  and `concavity_error` in particular — are strongly right-skewed with long upper
  tails. Centring on the median and scaling by the interquartile range keeps those
  tails from dominating the transform.
- **Median imputation** retained even though this dataset is complete, so the same
  pipeline survives a messier CSV unchanged.
- **One-hot encoding branch is built but not fitted**, since there are no
  categorical columns. The `ColumnTransformer` adds it automatically if you swap
  in a dataset that has them.
- **`class_weight="balanced"`** on Logistic Regression and Decision Tree,
  `"balanced_subsample"` on Random Forest. kNN and GaussianNB do not support it.
- Everything above sits inside a scikit-learn `Pipeline`, so cross-validation
  refits the preprocessing **inside each fold**. Fitting a scaler on the whole
  dataset before splitting leaks test-fold statistics into training and inflates
  the scores; this structure makes that mistake impossible rather than merely
  avoided by convention. There is a unit test asserting it.

---

## c. GitHub Repository Link

**Repository:** [ml-assignment-2](https://github.com/2025ac05832/ml-assignment-2)

**Live Streamlit App:** [ml-assignment-2](https://2025ac05832-ml-assignment-2.streamlit.app/)

```
ml-assignment-2/
├── app.py                       Streamlit application
├── run_experiment.py            CLI: cross-validates everything, writes results/
├── requirements.txt             five dependencies
├── README.md                    this file
├── test_data.csv                114-row held-out slice for the app
├── data/
│   └── wdbc.csv                 full dataset, 569 rows
├── model/
│   ├── __init__.py
│   ├── datasets.py              loading + constraint validation
│   ├── classifiers.py           preprocessing pipeline + the five estimators
│   ├── evaluation.py            cross-validated metrics
│   └── ML_Assignment2_Colab.ipynb
├── tests/
│   └── test_model.py            15 unit tests, standard-library unittest
└── results/
    ├── metrics_summary.csv
    ├── metrics_per_fold.csv
    ├── metric_spread.png
    └── confusion_matrices.png
```

---

## d. Models Used

### Evaluation methodology

Rather than a single 80/20 hold-out split, every model is scored with **repeated
stratified k-fold cross-validation** — 5 folds repeated 3 times, so 15 fits per
model — and reported as mean ± standard deviation.

The reason is the dataset size. A single 80/20 split puts about 114 rows in the
test set, and moving the random seed can shift accuracy by two or three
percentage points, which is enough to reorder the models. Reporting the spread
makes it visible when two models are genuinely tied rather than separated by
split luck. The comparison table below contains a clear example of exactly that.

### Comparison Table

Mean ± SD across 15 folds:

| ML Model Name            | Accuracy        | AUC             | Precision       | Recall          | F1              | MCC             |
| :----------------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- |
| Logistic Regression      | 0.9736 ± 0.0142 | 0.9945 ± 0.0048 | 0.9675 ± 0.0258 | 0.9622 ± 0.0279 | 0.9645 ± 0.0193 | 0.9439 ± 0.0304 |
| Decision Tree            | 0.9379 ± 0.0247 | 0.9641 ± 0.0209 | 0.9025 ± 0.0469 | 0.9372 ± 0.0382 | 0.9186 ± 0.0310 | 0.8700 ± 0.0509 |
| kNN                      | 0.9631 ± 0.0138 | 0.9894 ± 0.0075 | 0.9841 ± 0.0233 | 0.9167 ± 0.0417 | 0.9484 ± 0.0199 | 0.9221 ± 0.0291 |
| Naive Bayes              | 0.9315 ± 0.0182 | 0.9872 ± 0.0063 | 0.9190 ± 0.0393 | 0.8979 ± 0.0474 | 0.9069 ± 0.0249 | 0.8545 ± 0.0379 |
| Random Forest (Ensemble) | 0.9625 ± 0.0181 | 0.9908 ± 0.0082 | 0.9556 ± 0.0348 | 0.9449 ± 0.0441 | 0.9492 ± 0.0246 | 0.9208 ± 0.0381 |

Positive class for Precision / Recall / F1 is `malignant`.

Fold-by-fold spread: `results/metric_spread.png`.
Out-of-fold confusion matrices, in which every one of the 569 rows is predicted
exactly once by a model that did not train on it: `results/confusion_matrices.png`.

### Single-split figures (what the app reports)

For comparison, the same models fitted once on the 455 training rows and scored
on the 114-row `test_data.csv`:

| ML Model Name            | Accuracy |     F1 |    MCC | Misclassified |
| :----------------------- | -------: | -----: | -----: | ------------: |
| Logistic Regression      |   0.9737 | 0.9639 | 0.9433 |       3 / 114 |
| Decision Tree            |   0.9123 | 0.8780 | 0.8102 |      10 / 114 |
| kNN                      |   0.9386 | 0.9114 | 0.8688 |       7 / 114 |
| Naive Bayes              |   0.9211 | 0.8889 | 0.8292 |       9 / 114 |
| Random Forest (Ensemble) |   0.9737 | 0.9630 | 0.9442 |       3 / 114 |

### Observations

| ML Model Name            | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | **Top Overall Performer.** Achieved the highest mean on 5 of 6 metrics (Accuracy: **0.9736**, AUC: **0.9945**, Recall: **0.9622**, F1: **0.9645**, MCC: **0.9439**) with the tightest standard deviation on Accuracy (±0.0142). This demonstrates that the 30 WDBC features, once centered and scaled via `RobustScaler`, are largely linearly separable. A fast, highly interpretable linear model outperforming complex ensembles is a compelling result for clinical practice. |
| Decision Tree            | **Highest Variance and Lowest Generalization.** Ranked lowest across Accuracy (**0.9379**), AUC (**0.9641**), and MCC (**0.8700**), while exhibiting the widest cross-fold standard deviation (MCC SD ±0.0509). High variance across partitions is the classic signature of single unpruned/shallow trees due to rigid orthogonal splits, directly illustrating why ensemble bagging is required to stabilize decision boundaries.                                    |
| kNN                      | **High Precision, Sub-optimal Diagnostic Recall.** Delivered the highest precision across all models (**0.9841 ± 0.0233**), meaning it rarely triggers false alarms. However, it exhibited the second-lowest recall on malignant cases (**0.9167 ± 0.0417**). In clinical diagnostics where missing a malignant tumor (false negative) is far more dangerous than reviewing a benign mass (false positive), kNN's error profile is disadvantageous.                                 |
| Naive Bayes              | **Strong Instance Ranking despite Violated Independence.** Exhibited the lowest overall Accuracy (**0.9315**) and Recall (**0.8979**), yet sustained an impressive AUC (**0.9872 ± 0.0063**). The feature set has severe multicollinearity (e.g., `mean_radius`, `radius_error`, and `worst_radius` are correlated views of one attribute), violating conditional independence. While distorted probability scales hurt discrete thresholding, relative ranking remains strong.      |
| Random Forest (Ensemble) | **Substantial Variance Reduction over Single Tree.** Gained +2.46% accuracy and +5.08% MCC over the single Decision Tree, while reducing variance by over 25% (MCC SD dropped to ±0.0381). Bootstrap aggregation and random feature subsampling successfully mitigate individual tree instability, though it does not surpass the linear boundary established by Logistic Regression.                                                                                    |
| **Overall Winner**       | **Logistic Regression.** It leads across all critical clinical metrics with unmatched consistency. Furthermore, repeated cross-validation surfaces an important insight: kNN (F1: 0.9484 ± 0.0199) and Random Forest (F1: 0.9492 ± 0.0246) differ by only 0.0008, well within error margins and thus statistically tied—a nuance that single holdout splits fail to capture.
---

## Streamlit App Features

| Requirement                              | Implementation                                                                                                                                                                                        |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset upload option (CSV)              | Sidebar file uploader, with the bundled 114-row `test_data.csv` as fallback                                                                                                                           |
| Model selection dropdown                 | Sidebar `selectbox` across all five pipelines                                                                                                                                                         |
| Display of evaluation metrics            | "Evaluate on a CSV" tab — all six metrics as metric cards                                                                                                                                             |
| Confusion matrix / classification report | Same tab, heatmap and per-class report side by side                                                                                                                                                   |
| Additional                               | Cross-validation tab with adjustable folds and repeats and a live box plot of the spread; a mistakes-only filter on the predictions table; a dataset tab with per-feature class-separation histograms |

The app fits the models on the **training rows only**, reproducing the same
stratified split and seed used by `run_experiment.py`, so the rows in
`test_data.csv` are genuinely unseen. Fitting on the full frame would have the app
score itself on data it had already trained on and report a meaningless 1.0000.

---

## How to Run Locally

```bash
git clone https://github.com/2025ac05832/ml-assignment-2.git
cd ml-assignment-2
pip install -r requirements.txt

python -m unittest discover -s tests -v   # 15 tests, should all pass
python run_experiment.py                  # writes results/
streamlit run app.py
```

To use a different dataset:

```bash
python run_experiment.py --csv data/your_data.csv --target your_column --folds 10
```

`model/datasets.py` validates the ≥12 feature and ≥500 instance constraints and
raises a specific `DatasetError` rather than failing three frames deep.

---

## BITS Virtual Lab Execution

`model/ML_Assignment2_Colab.ipynb` is the notebook version of the pipeline. A
screenshot of it executing on BITS Virtual Lab is included in the submission PDF.

---

## Notes on Dependencies and Reproducibility

This version deliberately minimises external dependencies:

- **Five packages**: `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`.
  No `seaborn` (matplotlib covers every plot), no `joblib` (nothing is pickled).
- **No dataset download.** WDBC ships inside scikit-learn; `data/wdbc.csv` is the
  committed copy actually used.
- **No binary artifacts in the repository.** The app rebuilds the models from
  source at startup, cached with `st.cache_resource`. A full refit of all five
  takes about two seconds on 569 rows. This removes the most common cause of a
  Streamlit Cloud deployment dying on first boot — a pickle written by one
  scikit-learn version failing to load under another — and means the app can
  never drift out of sync with the training code.
- **No pytest.** The test suite uses the standard library's `unittest`.
- `random_state=42` throughout. Re-running `run_experiment.py` reproduces every
  number above exactly.
