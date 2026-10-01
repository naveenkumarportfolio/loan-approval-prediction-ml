# =============================================================================
# Loan Approval Prediction: Decision Tree vs Naive Bayes vs KNN
# =============================================================================
# Required libraries (scikit-learn 1.2 or newer recommended):
# pandas
# numpy
# matplotlib
# seaborn
# scikit-learn
#
# Place Loan_Data.csv in the SAME folder as this script, then run it.
# =============================================================================

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts to files without opening pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
RANDOM_STATE = 42   # Fixed so results are reproducible
TEST_SIZE = 0.20    # 80% training, 20% testing

# Folder of this script (falls back to current folder if run interactively)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_FILE = BASE_DIR / "Loan_Data.csv"
OUTPUT_DIR = BASE_DIR / "output"

# Actual columns found in Loan_Data.csv (checked by inspecting the file):
TARGET_COLUMN = "Loan_Status"   # Values: 'Y' = loan approved, 'N' = loan rejected
ID_COLUMN = "Loan_ID"           # Unique identifier, carries no predictive meaning


# -----------------------------------------------------------------------------
# STEP 1: LOAD AND INSPECT DATA
# -----------------------------------------------------------------------------
def load_data(path):
    """Read the CSV file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path.name}'. Put it in the same folder as this script: {path.parent}"
        )
    return pd.read_csv(path)


def show_basic_info(df):
    """Print shape, columns, data types, missing values, duplicates, target distribution."""
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Dataset shape (rows, columns): {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes.to_string())

    missing = df.isnull().sum()
    print("\nMissing-value summary (columns with missing values only):")
    if missing.sum() == 0:
        print("No missing values.")
    else:
        summary = pd.DataFrame({
            "Missing": missing[missing > 0],
            "Percent": (missing[missing > 0] / len(df) * 100).round(2),
        })
        print(summary.to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print(f"\nTarget distribution ({TARGET_COLUMN}):")
    counts = df[TARGET_COLUMN].value_counts()
    percent = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2)
    print(pd.DataFrame({"Count": counts, "Percent": percent}).to_string())


# -----------------------------------------------------------------------------
# STEP 2: PREPARE DATA
# -----------------------------------------------------------------------------
def prepare_data(df):
    """
    Clean the data and split it into features (X) and target (y).
    Imputing/scaling/encoding is NOT done here; it happens inside the Pipeline
    so it is fitted only on the training data (no data leakage).
    """
    df = df.copy()

    # Remove duplicate rows (if any)
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nRows removed as duplicates: {before - len(df)}")

    # Drop the identifier column (it only labels each applicant)
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
        print(f"Dropped identifier column: {ID_COLUMN}")

    # Encode the target: Y (approved) -> 1, N (rejected) -> 0
    y = df[TARGET_COLUMN].map({"Y": 1, "N": 0})
    X = df.drop(columns=[TARGET_COLUMN])

    # Credit_History is stored as 0.0/1.0 but really means "bad/good record",
    # so treat it as a category instead of a number.
    X["Credit_History"] = X["Credit_History"].map({1.0: "Good", 0.0: "Bad"})

    # Detect numeric and categorical columns automatically
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print(f"Numerical columns  : {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """
    Numeric columns:     fill missing with median, then standardise (mean 0, std 1).
    Categorical columns: fill missing with most frequent value, then one-hot encode.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


# -----------------------------------------------------------------------------
# STEP 3: MODELS
# -----------------------------------------------------------------------------
def build_models(numeric_cols, categorical_cols):
    """
    Create the three models. Each is a Pipeline (preprocessing + classifier).
    - Decision Tree: depth limited to reduce overfitting and keep it explainable.
    - Naive Bayes: GaussianNB, because the processed data is numeric
      (scaled numbers + 0/1 encoded categories).
    - KNN: k = 5 neighbours; scaling (done in preprocessing) is essential for it.
    """
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    return {
        name: Pipeline([
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", clf),
        ])
        for name, clf in models.items()
    }


# -----------------------------------------------------------------------------
# STEP 4: EVALUATION
# -----------------------------------------------------------------------------
# Class meaning in this dataset:
#   1 = Loan APPROVED (Y)  <- "positive" class used for Precision/Recall/F1
#   0 = Loan REJECTED (N)
#
# FALSE POSITIVE (FP): model predicts APPROVED, but the application was actually
#   REJECTED. The bank may lend to a risky borrower -> possible default and
#   financial loss. Usually the more costly error for a bank.
# FALSE NEGATIVE (FN): model predicts REJECTED, but the application was actually
#   APPROVED. A creditworthy customer is turned away -> lost interest income,
#   lost customer, possible unfairness complaints.
#
# Precision (approved) = of loans the model approved, how many were truly approved
#                        (low precision = many risky approvals).
# Recall (approved)    = of loans that should be approved, how many the model found
#                        (low recall = many good customers rejected).
def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """Train one model, predict on the test set, and return its metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall (Rejected)": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print("=" * 70)
    print(f"Accuracy                       : {results['Accuracy']:.4f}")
    print(f"Precision (Approved)           : {results['Precision']:.4f}")
    print(f"Recall (Approved)              : {results['Recall']:.4f}")
    print(f"F1-score (Approved)            : {results['F1-score']:.4f}")
    print(f"Recall (Rejected)              : {results['Recall (Rejected)']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Rejected (N)", "Approved (Y)"], zero_division=0,
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(
        cm,
        index=["Actual Rejected", "Actual Approved"],
        columns=["Pred Rejected", "Pred Approved"],
    ).to_string())
    print(f"\nFalse Positives (risky loans wrongly approved)     : {fp}")
    print(f"False Negatives (good applicants wrongly rejected) : {fn}")

    return results, cm


# -----------------------------------------------------------------------------
# STEP 5: PLOTS
# -----------------------------------------------------------------------------
def plot_confusion_matrix(cm, model_name, filename):
    """Save a confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
    )
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def plot_metric_comparison(results_df, metric, filename):
    """Save a bar chart comparing one metric across the three models."""
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=results_df, x="Model", y=metric,
        hue="Model", palette="viridis", legend=False,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    plt.ylim(0, 1.1)
    title = metric if metric == "Accuracy" else f"{metric} (Approved class)"
    plt.title(f"Model {title} Comparison")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# STEP 6: DATA-DRIVEN SUMMARY AND MODEL DEFENCE
# -----------------------------------------------------------------------------
def print_summary(results_df):
    """Print a comparison built only from the calculated results."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE (positive class = Loan Approved)")
    print("=" * 70)
    table = results_df[[
        "Model", "Accuracy", "Precision", "Recall", "F1-score", "FP", "FN"
    ]].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        table[col] = table[col].round(4)
    print(table.to_string(index=False))

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY (based on the calculated results)")
    print("=" * 70)

    for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
        top = results_df.loc[results_df[metric].idxmax()]
        print(f"- Highest {metric:<9}: {top['Model']} ({top[metric]:.4f})")

    fewest_fp = results_df.loc[results_df["FP"].idxmin()]
    fewest_fn = results_df.loc[results_df["FN"].idxmin()]
    print(f"- Fewest false positives (risky approvals) : {fewest_fp['Model']} ({int(fewest_fp['FP'])})")
    print(f"- Fewest false negatives (missed good loans): {fewest_fn['Model']} ({int(fewest_fn['FN'])})")

    print("\nWhy the two error types matter for a bank:")
    print("- False approval (FP): money is lent to someone who should have been rejected,")
    print("  so the bank can lose principal and interest through default.")
    print("- False rejection (FN): a creditworthy customer is turned away, so the bank")
    print("  loses income and goodwill, and the customer may feel treated unfairly.")
    print("- Which error is worse depends on the bank's risk appetite, so precision and")
    print("  recall are both considered instead of accuracy alone.")

    print("\nExplainability (a property of each algorithm type, not a calculated value):")
    print("- Decision Tree: decisions follow if-then rules that can be read out and")
    print("  explained to a customer (e.g. which factor led to rejection).")
    print("- Naive Bayes: based on class probabilities per feature; explainable in")
    print("  general terms but harder to turn into a simple reason for one customer.")
    print("- KNN: decides by similarity to past applicants; offers no clear rules or")
    print("  feature weights, so it is the hardest to justify to a customer.")

    # Choose the best model: highest F1-score for the approved class
    # (balances precision and recall); accuracy breaks ties.
    ranked = results_df.sort_values(["F1-score", "Accuracy"], ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    print("\n" + "=" * 70)
    print("BEST MODEL ACCORDING TO THE CALCULATED RESULTS")
    print("=" * 70)
    print(f"Selection rule: highest F1-score for the Approved class (accuracy breaks ties).")
    print(f"Best model: {best['Model']}")
    print(f"  Accuracy={best['Accuracy']:.4f}, Precision={best['Precision']:.4f}, "
          f"Recall={best['Recall']:.4f}, F1={best['F1-score']:.4f}")
    print(f"  False positives={int(best['FP'])}, False negatives={int(best['FN'])}")
    if best["Model"] != fewest_fp["Model"]:
        print(f"  Note: {fewest_fp['Model']} made fewer false approvals ({int(fewest_fp['FP'])}), "
              f"so a bank focused only on avoiding bad loans might weigh it differently.")
    print("  Results come from a single train/test split, so small differences between")
    print("  models may change with a different split.")


# -----------------------------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_FILE)
    show_basic_info(df)

    X, y, numeric_cols, categorical_cols = prepare_data(df)

    # Same stratified split for all three models (fair comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)
    print(f"Training set size: {X_train.shape[0]} rows ({X_train.shape[1]} features)")
    print(f"Testing set size : {X_test.shape[0]} rows ({X_test.shape[1]} features)")
    print("Class meaning: 1 = Loan Approved (Y), 0 = Loan Rejected (N)")

    models = build_models(numeric_cols, categorical_cols)

    all_results = []
    cm_files = {
        "Decision Tree": "decision_tree_confusion_matrix.png",
        "Naive Bayes": "naive_bayes_confusion_matrix.png",
        "KNN": "knn_confusion_matrix.png",
    }
    for name, pipeline in models.items():
        results, cm = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        all_results.append(results)
        plot_confusion_matrix(cm, name, cm_files[name])

    results_df = pd.DataFrame(all_results)

    plot_metric_comparison(results_df, "Accuracy", "accuracy_comparison.png")
    plot_metric_comparison(results_df, "Precision", "precision_comparison.png")
    plot_metric_comparison(results_df, "Recall", "recall_comparison.png")
    plot_metric_comparison(results_df, "F1-score", "f1_comparison.png")

    print_summary(results_df)
    print(f"\nAll charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()