import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_validate,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "Student_performance_data.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

CLASSIFIER_PATH = os.path.join(
    MODELS_DIR,
    "academic_performance_model.pkl"
)

REGRESSOR_PATH = os.path.join(
    MODELS_DIR,
    "academic_gpa_model.pkl"
)

FEATURE_PATH = os.path.join(
    MODELS_DIR,
    "feature_names.pkl"
)

RESULTS_PATH = os.path.join(
    MODELS_DIR,
    "model_evaluation_results.json"
)

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# APPROVED ML FEATURES
# ============================================================

FEATURE_NAMES = [
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering"
]

CLASSIFICATION_TARGET = "GradeClass"
REGRESSION_TARGET = "GPA"

VALID_GRADE_CLASSES = {0, 1, 2, 3, 4}


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_training_data():

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = (
        FEATURE_NAMES
        + [
            "StudentID",
            "GPA",
            "GradeClass"
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if len(df) < 20:
        raise ValueError(
            "Dataset must contain at least 20 rows."
        )

    # Convert model inputs and targets to numeric.
    numeric_columns = FEATURE_NAMES + [
        "GPA",
        "GradeClass"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove incomplete training rows.
    df = df.dropna(
        subset=numeric_columns
    ).copy()

    if len(df) < 20:
        raise ValueError(
            "Fewer than 20 valid rows remain after "
            "removing missing values."
        )

    # Ensure all values are finite.
    if not np.isfinite(
        df[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Dataset contains infinite numeric values."
        )

    # GPA scale validation.
    if not df["GPA"].between(1, 10).all():
        raise ValueError(
            "GPA must be between 1 and 10. "
            "Check your CSV before training."
        )

    # GradeClass validation.
    actual_classes = set(
        df["GradeClass"].unique().tolist()
    )

    if not actual_classes.issubset(
        VALID_GRADE_CLASSES
    ):
        raise ValueError(
            "GradeClass must contain only 0, 1, 2, 3, 4."
        )

    df["GradeClass"] = df["GradeClass"].astype(int)

    if df["GradeClass"].nunique() < 2:
        raise ValueError(
            "At least two GradeClass categories are required."
        )

    # Ensure each class has enough samples for 5-fold CV.
    class_counts = df["GradeClass"].value_counts()

    if class_counts.min() < 5:
        raise ValueError(
            "Every GradeClass represented in the dataset "
            "must have at least 5 rows for 5-fold "
            "stratified cross-validation."
        )

    # Prevent data leakage.
    forbidden_features = {
        "StudentID",
        "GPA",
        "GradeClass"
    }

    if forbidden_features.intersection(FEATURE_NAMES):
        raise ValueError(
            "Data leakage detected. StudentID, GPA, and "
            "GradeClass must not be model input features."
        )

    return df


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

def get_classification_models():

    return {
        "Logistic Regression": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                ))
            ]),
            "params": {
                "model__C": [
                    0.1, 0.5, 1.0, 2.0, 5.0
                ]
            },
            "iterations": 5
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=1
            ),
            "params": {
                "n_estimators": [150, 250, 350],
                "max_depth": [None, 10, 15, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"]
            },
            "iterations": 8
        },

        "Extra Trees": {
            "model": ExtraTreesClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=1
            ),
            "params": {
                "n_estimators": [150, 250, 350],
                "max_depth": [None, 10, 15, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"]
            },
            "iterations": 8
        },

        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                random_state=42
            ),
            "params": {
                "n_estimators": [75, 100, 150, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0]
            },
            "iterations": 8
        }
    }


# ============================================================
# REGRESSION MODELS
# ============================================================

def get_regression_models():

    return {
        "Linear Regression": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            "params": {},
            "iterations": 1
        },

        "Random Forest Regressor": {
            "model": RandomForestRegressor(
                random_state=42,
                n_jobs=1
            ),
            "params": {
                "n_estimators": [150, 250, 350],
                "max_depth": [None, 10, 15, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": [1.0, "sqrt"]
            },
            "iterations": 8
        },

        "Extra Trees Regressor": {
            "model": ExtraTreesRegressor(
                random_state=42,
                n_jobs=1
            ),
            "params": {
                "n_estimators": [150, 250, 350],
                "max_depth": [None, 10, 15, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": [1.0, "sqrt"]
            },
            "iterations": 8
        },

        "Gradient Boosting Regressor": {
            "model": GradientBoostingRegressor(
                random_state=42
            ),
            "params": {
                "n_estimators": [75, 100, 150, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0]
            },
            "iterations": 8
        }
    }


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def tune_classifier(name, config, X, y):

    model = config["model"]
    parameters = config["params"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    if not parameters:
        model.fit(X, y)
        return model, {}, None

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=config["iterations"],
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=1,
        refit=True,
        verbose=0
    )

    search.fit(X, y)

    return (
        search.best_estimator_,
        search.best_params_,
        float(search.best_score_)
    )


def tune_regressor(name, config, X, y):

    model = config["model"]
    parameters = config["params"]

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    if not parameters:
        model.fit(X, y)
        return model, {}, None

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=config["iterations"],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=1,
        refit=True,
        verbose=0
    )

    search.fit(X, y)

    return (
        search.best_estimator_,
        search.best_params_,
        float(search.best_score_)
    )


# ============================================================
# CLASSIFICATION CROSS-VALIDATION
# ============================================================

def classification_cross_validation(model, X, y):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
            "weighted_f1": "f1_weighted"
        },
        n_jobs=1,
        return_train_score=False
    )

    return {
        "cv_accuracy": float(
            np.mean(results["test_accuracy"])
        ),
        "cv_accuracy_std": float(
            np.std(results["test_accuracy"])
        ),
        "cv_balanced_accuracy": float(
            np.mean(results["test_balanced_accuracy"])
        ),
        "cv_macro_f1": float(
            np.mean(results["test_macro_f1"])
        ),
        "cv_weighted_f1": float(
            np.mean(results["test_weighted_f1"])
        )
    }


# ============================================================
# REGRESSION CROSS-VALIDATION
# ============================================================

def regression_cross_validation(model, X, y):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2"
        },
        n_jobs=1,
        return_train_score=False
    )

    cv_mae = -np.mean(results["test_mae"])
    cv_mse = -np.mean(results["test_mse"])

    return {
        "cv_mae": float(cv_mae),
        "cv_mse": float(cv_mse),
        "cv_rmse": float(np.sqrt(max(0.0, cv_mse))),
        "cv_r2": float(np.mean(results["test_r2"]))
    }


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classifier(model, X, y):

    predictions = model.predict(X)

    labels = sorted(y.unique().tolist())

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "macro_precision": float(
            precision_score(
                y, predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y, predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y, predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y, predictions,
                average="weighted",
                zero_division=0
            )
        ),
        "per_class_precision": [
            float(value)
            for value in precision_score(
                y, predictions,
                labels=labels,
                average=None,
                zero_division=0
            )
        ],
        "per_class_recall": [
            float(value)
            for value in recall_score(
                y, predictions,
                labels=labels,
                average=None,
                zero_division=0
            )
        ],
        "per_class_f1": [
            float(value)
            for value in f1_score(
                y, predictions,
                labels=labels,
                average=None,
                zero_division=0
            )
        ],
        "confusion_matrix": confusion_matrix(
            y, predictions,
            labels=labels
        ).tolist(),
        "class_labels": [
            int(label) for label in labels
        ]
    }


# ============================================================
# REGRESSION EVALUATION
# ============================================================

def evaluate_regressor(model, X, y):

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)

    return {
        "mae": float(
            mean_absolute_error(y, predictions)
        ),
        "mse": float(mse),
        "rmse": float(np.sqrt(max(0.0, mse))),
        "r2": float(
            r2_score(y, predictions)
        )
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(model, feature_names):

    estimator = model

    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_

    elif hasattr(estimator, "coef_"):
        coefficients = np.abs(estimator.coef_)

        if coefficients.ndim == 2:
            values = np.mean(coefficients, axis=0)
        else:
            values = coefficients

    else:
        return {}

    return {
        feature: float(value)
        for feature, value in zip(feature_names, values)
    }


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models():

    print("\n" + "=" * 70)
    print("ACADEMIC PERFORMANCE ANALYTICS")
    print("GPA SCALE: 1-10")
    print("=" * 70)

    df = load_training_data()

    print(f"\nDataset shape: {df.shape}")
    print(f"Model features: {len(FEATURE_NAMES)}")
    print("StudentID excluded: YES")
    print("GPA leakage into GradeClass: NO")

    X = df[FEATURE_NAMES].copy()
    y_classification = df[CLASSIFICATION_TARGET].copy()
    y_regression = df[REGRESSION_TARGET].copy()

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASSIFICATION MODEL TUNING")
    print("-" * 70)

    classification_results = {}

    best_classifier = None
    best_classifier_name = None
    best_classifier_score = -np.inf

    for name, config in get_classification_models().items():

        print(f"\nTraining: {name}")

        model, best_params, tuning_score = tune_classifier(
            name,
            config,
            X,
            y_classification
        )

        cv_metrics = classification_cross_validation(
            model,
            X,
            y_classification
        )

        train_metrics = evaluate_classifier(
            model,
            X,
            y_classification
        )

        combined = {
            **cv_metrics,
            **train_metrics,
            "best_params": best_params,
            "tuning_score": tuning_score
        }

        classification_results[name] = combined

        print(
            f"CV Accuracy: {cv_metrics['cv_accuracy']:.4f}"
        )
        print(
            f"CV Macro F1: {cv_metrics['cv_macro_f1']:.4f}"
        )
        print(
            f"CV Balanced Accuracy: "
            f"{cv_metrics['cv_balanced_accuracy']:.4f}"
        )

        score = cv_metrics["cv_macro_f1"]

        if score > best_classifier_score:
            best_classifier_score = score
            best_classifier = model
            best_classifier_name = name

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("GPA REGRESSION MODEL TUNING")
    print("-" * 70)

    regression_results = {}

    best_regressor = None
    best_regressor_name = None
    best_regressor_score = np.inf

    for name, config in get_regression_models().items():

        print(f"\nTraining: {name}")

        model, best_params, tuning_score = tune_regressor(
            name,
            config,
            X,
            y_regression
        )

        cv_metrics = regression_cross_validation(
            model,
            X,
            y_regression
        )

        train_metrics = evaluate_regressor(
            model,
            X,
            y_regression
        )

        combined = {
            **cv_metrics,
            **train_metrics,
            "best_params": best_params,
            "tuning_score": tuning_score
        }

        regression_results[name] = combined

        print(
            f"CV MAE: {cv_metrics['cv_mae']:.4f}"
        )
        print(
            f"CV RMSE: {cv_metrics['cv_rmse']:.4f}"
        )
        print(
            f"CV R2: {cv_metrics['cv_r2']:.4f}"
        )

        score = cv_metrics["cv_rmse"]

        if score < best_regressor_score:
            best_regressor_score = score
            best_regressor = model
            best_regressor_name = name

    if best_classifier is None or best_regressor is None:
        raise RuntimeError(
            "Training did not produce valid models."
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = get_feature_importance(
        best_classifier,
        FEATURE_NAMES
    )

    feature_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # --------------------------------------------------------
    # Save trained models
    # --------------------------------------------------------

    joblib.dump(
        best_classifier,
        CLASSIFIER_PATH
    )

    joblib.dump(
        best_regressor,
        REGRESSOR_PATH
    )

    joblib.dump(
        FEATURE_NAMES,
        FEATURE_PATH
    )

    # --------------------------------------------------------
    # Evaluation report
    # --------------------------------------------------------

    best_classifier_metrics = classification_results[
        best_classifier_name
    ]

    evaluation_results = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "features": FEATURE_NAMES,
            "classification_target": CLASSIFICATION_TARGET,
            "regression_target": REGRESSION_TARGET
        },

        "gpa_scale": "1-10",

        "data_leakage_protection": {
            "student_id_excluded": True,
            "gpa_excluded_from_gradeclass": True,
            "gradeclass_excluded_from_features": True
        },

        "best_classification_model": best_classifier_name,
        "best_regression_model": best_regressor_name,

        # Aliases for compatibility with earlier scripts.
        "best_classifier": best_classifier_name,
        "best_regressor": best_regressor_name,

        "best_classifier_macro_f1": float(
            best_classifier_score
        ),

        "best_regressor_mae": float(
            regression_results[best_regressor_name]["cv_mae"]
        ),

        "classification_models": classification_results,
        "regression_models": regression_results,

        "feature_importance": feature_importance,

        "confusion_matrix": best_classifier_metrics[
            "confusion_matrix"
        ],

        "class_labels": best_classifier_metrics[
            "class_labels"
        ]
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            evaluation_results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(f"Best classifier: {best_classifier_name}")
    print(
        f"CV Macro F1: {best_classifier_score:.4f}"
    )

    print(f"Best regressor: {best_regressor_name}")
    print(
        f"CV RMSE: {best_regressor_score:.4f}"
    )

    print("\nSaved model files:")
    print(CLASSIFIER_PATH)
    print(REGRESSOR_PATH)
    print(FEATURE_PATH)
    print(RESULTS_PATH)

    print("=" * 70)

    return evaluation_results


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    try:
        train_all_models()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    except Exception as error:
        print("\nTRAINING ERROR")
        print(type(error).__name__ + ":", str(error))
        raise