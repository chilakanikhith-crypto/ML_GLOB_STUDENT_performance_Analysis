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

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

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
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "Student_performance_data.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


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


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():

    if not os.path.exists(CSV_PATH):

        raise FileNotFoundError(
            f"Dataset not found:\n{CSV_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH
    )

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

    # Convert features to numeric
    for column in FEATURE_NAMES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["GPA"] = pd.to_numeric(
        df["GPA"],
        errors="coerce"
    )

    df["GradeClass"] = pd.to_numeric(
        df["GradeClass"],
        errors="coerce"
    )

    required_for_training = (
        FEATURE_NAMES
        + [
            "GPA",
            "GradeClass"
        ]
    )

    df = df.dropna(
        subset=required_for_training
    ).copy()

    if len(df) < 20:

        raise ValueError(
            "Not enough valid rows after removing missing values."
        )

    # ========================================================
    # DATA LEAKAGE PROTECTION
    # ========================================================

    forbidden_features = {
        "StudentID",
        "GPA",
        "GradeClass"
    }

    if forbidden_features.intersection(
        set(FEATURE_NAMES)
    ):

        raise ValueError(
            "Data leakage detected. "
            "StudentID, GPA and GradeClass "
            "cannot be model features."
        )

    return df


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

def get_classification_models():

    models = {

        "Logistic Regression": {

            "model": Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=42
                        )
                    )
                ]
            ),

            "params": {

                "model__C": [
                    0.1,
                    0.5,
                    1.0,
                    2.0,
                    5.0
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

                "n_estimators": [
                    150,
                    250,
                    350
                ],

                "max_depth": [
                    None,
                    10,
                    15,
                    20
                ],

                "min_samples_split": [
                    2,
                    5
                ],

                "min_samples_leaf": [
                    1,
                    2
                ],

                "max_features": [
                    "sqrt",
                    "log2"
                ]

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

                "n_estimators": [
                    150,
                    250,
                    350
                ],

                "max_depth": [
                    None,
                    10,
                    15,
                    20
                ],

                "min_samples_split": [
                    2,
                    5
                ],

                "min_samples_leaf": [
                    1,
                    2
                ],

                "max_features": [
                    "sqrt",
                    "log2"
                ]

            },

            "iterations": 8

        },


        "Gradient Boosting": {

            "model": GradientBoostingClassifier(
                random_state=42
            ),

            "params": {

                "n_estimators": [
                    75,
                    100,
                    150,
                    200
                ],

                "learning_rate": [
                    0.03,
                    0.05,
                    0.1
                ],

                "max_depth": [
                    2,
                    3,
                    4
                ],

                "subsample": [
                    0.8,
                    1.0
                ]

            },

            "iterations": 8

        }

    }

    return models


# ============================================================
# REGRESSION MODELS
# ============================================================

def get_regression_models():

    models = {

        "Linear Regression": {

            "model": Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "model",
                        LinearRegression()
                    )
                ]
            ),

            "params": {},

            "iterations": 1

        },


        "Random Forest Regressor": {

            "model": RandomForestRegressor(
                random_state=42,
                n_jobs=1
            ),

            "params": {

                "n_estimators": [
                    150,
                    250,
                    350
                ],

                "max_depth": [
                    None,
                    10,
                    15,
                    20
                ],

                "min_samples_split": [
                    2,
                    5
                ],

                "min_samples_leaf": [
                    1,
                    2
                ],

                "max_features": [
                    1.0,
                    "sqrt"
                ]

            },

            "iterations": 8

        },


        "Extra Trees Regressor": {

            "model": ExtraTreesRegressor(
                random_state=42,
                n_jobs=1
            ),

            "params": {

                "n_estimators": [
                    150,
                    250,
                    350
                ],

                "max_depth": [
                    None,
                    10,
                    15,
                    20
                ],

                "min_samples_split": [
                    2,
                    5
                ],

                "min_samples_leaf": [
                    1,
                    2
                ],

                "max_features": [
                    1.0,
                    "sqrt"
                ]

            },

            "iterations": 8

        },


        "Gradient Boosting Regressor": {

            "model": GradientBoostingRegressor(
                random_state=42
            ),

            "params": {

                "n_estimators": [
                    75,
                    100,
                    150,
                    200
                ],

                "learning_rate": [
                    0.03,
                    0.05,
                    0.1
                ],

                "max_depth": [
                    2,
                    3,
                    4
                ],

                "subsample": [
                    0.8,
                    1.0
                ]

            },

            "iterations": 8

        }

    }

    return models


# ============================================================
# CLASSIFICATION CROSS VALIDATION
# ============================================================

def classification_cross_validation(
    model,
    X,
    y
):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {

        "accuracy":
            "accuracy",

        "balanced_accuracy":
            "balanced_accuracy",

        "macro_f1":
            "f1_macro",

        "weighted_f1":
            "f1_weighted"

    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    return {

        "cv_accuracy": float(
            np.mean(
                results[
                    "test_accuracy"
                ]
            )
        ),

        "cv_accuracy_std": float(
            np.std(
                results[
                    "test_accuracy"
                ]
            )
        ),

        "cv_balanced_accuracy": float(
            np.mean(
                results[
                    "test_balanced_accuracy"
                ]
            )
        ),

        "cv_macro_f1": float(
            np.mean(
                results[
                    "test_macro_f1"
                ]
            )
        ),

        "cv_weighted_f1": float(
            np.mean(
                results[
                    "test_weighted_f1"
                ]
            )
        )

    }


# ============================================================
# REGRESSION CROSS VALIDATION
# ============================================================

def regression_cross_validation(
    model,
    X,
    y
):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {

        "mae":
            "neg_mean_absolute_error",

        "mse":
            "neg_mean_squared_error",

        "r2":
            "r2"

    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    cv_mae = -np.mean(
        results[
            "test_mae"
        ]
    )

    cv_mse = -np.mean(
        results[
            "test_mse"
        ]
    )

    cv_rmse = np.sqrt(
        cv_mse
    )

    cv_r2 = np.mean(
        results[
            "test_r2"
        ]
    )

    return {

        "cv_mae": float(
            cv_mae
        ),

        "cv_mse": float(
            cv_mse
        ),

        "cv_rmse": float(
            cv_rmse
        ),

        "cv_r2": float(
            cv_r2
        )

    }


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classifier(
    model,
    X,
    y
):

    predictions = model.predict(
        X
    )

    metrics = {

        "accuracy": float(
            accuracy_score(
                y,
                predictions
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                predictions
            )
        ),

        "macro_precision": float(
            precision_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "macro_recall": float(
            recall_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "weighted_f1": float(
            f1_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )
        )

    }

    labels = sorted(
        pd.Series(
            y
        ).unique().tolist()
    )

    metrics[
        "per_class_precision"
    ] = [
        float(value)
        for value in precision_score(
            y,
            predictions,
            labels=labels,
            average=None,
            zero_division=0
        )
    ]

    metrics[
        "per_class_recall"
    ] = [
        float(value)
        for value in recall_score(
            y,
            predictions,
            labels=labels,
            average=None,
            zero_division=0
        )
    ]

    metrics[
        "per_class_f1"
    ] = [
        float(value)
        for value in f1_score(
            y,
            predictions,
            labels=labels,
            average=None,
            zero_division=0
        )
    ]

    metrics[
        "confusion_matrix"
    ] = (
        confusion_matrix(
            y,
            predictions,
            labels=labels
        ).tolist()
    )

    metrics[
        "class_labels"
    ] = [

        float(label)
        if isinstance(
            label,
            (
                int,
                float,
                np.integer,
                np.floating
            )
        )
        else str(label)

        for label in labels

    ]

    return metrics


# ============================================================
# REGRESSION EVALUATION
# ============================================================

def evaluate_regressor(
    model,
    X,
    y
):

    predictions = model.predict(
        X
    )

    mse = mean_squared_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mse
    )

    return {

        "mae": float(
            mean_absolute_error(
                y,
                predictions
            )
        ),

        "mse": float(
            mse
        ),

        "rmse": float(
            rmse
        ),

        "r2": float(
            r2_score(
                y,
                predictions
            )
        )

    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model,
    feature_names
):

    estimator = model

    if isinstance(
        model,
        Pipeline
    ):

        estimator = model.named_steps[
            "model"
        ]

    # Tree models
    if hasattr(
        estimator,
        "feature_importances_"
    ):

        values = (
            estimator.feature_importances_
        )

        return {

            feature: float(
                value
            )

            for feature, value
            in zip(
                feature_names,
                values
            )

        }

    # Logistic Regression
    if hasattr(
        estimator,
        "coef_"
    ):

        coefficients = np.abs(
            estimator.coef_
        )

        if coefficients.ndim == 2:

            values = np.mean(
                coefficients,
                axis=0
            )

        else:

            values = coefficients

        return {

            feature: float(
                value
            )

            for feature, value
            in zip(
                feature_names,
                values
            )

        }

    return {}


# ============================================================
# TUNE CLASSIFIER
# ============================================================

def tune_classifier(
    name,
    config,
    X,
    y
):

    model = config[
        "model"
    ]

    parameters = config[
        "params"
    ]

    iterations = config[
        "iterations"
    ]

    if not parameters:

        model.fit(
            X,
            y
        )

        return (
            model,
            {},
            None
        )

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=parameters,

        n_iter=iterations,

        scoring="f1_macro",

        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        ),

        random_state=42,

        n_jobs=1,

        refit=True,

        verbose=0

    )

    search.fit(
        X,
        y
    )

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_
    )


# ============================================================
# TUNE REGRESSOR
# ============================================================

def tune_regressor(
    name,
    config,
    X,
    y
):

    model = config[
        "model"
    ]

    parameters = config[
        "params"
    ]

    iterations = config[
        "iterations"
    ]

    if not parameters:

        model.fit(
            X,
            y
        )

        return (
            model,
            {},
            None
        )

    search = RandomizedSearchCV(

        estimator=model,

        param_distributions=parameters,

        n_iter=iterations,

        scoring="neg_root_mean_squared_error",

        cv=KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        ),

        random_state=42,

        n_jobs=1,

        refit=True,

        verbose=0

    )

    search.fit(
        X,
        y
    )

    return (
        search.best_estimator_,
        search.best_params_,
        search.best_score_
    )


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models():

    print()
    print("=" * 70)
    print("ACADEMIC PERFORMANCE ANALYTICS")
    print("FAST HYPERPARAMETER TUNING")
    print("=" * 70)
    print()

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_training_data()

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Model features: "
        f"{len(FEATURE_NAMES)}"
    )

    print(
        "StudentID excluded: YES"
    )

    print(
        "GPA leakage into GradeClass: NO"
    )

    print()

    X = df[
        FEATURE_NAMES
    ].copy()

    y_classification = df[
        CLASSIFICATION_TARGET
    ].copy()

    y_regression = df[
        REGRESSION_TARGET
    ].copy()

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    print()
    print("-" * 70)
    print("CLASSIFICATION MODEL TUNING")
    print("-" * 70)

    classification_models = (
        get_classification_models()
    )

    classification_results = {}

    best_classifier = None

    best_classifier_name = None

    best_classifier_score = -np.inf

    for name, config in (
        classification_models.items()
    ):

        print()
        print(
            f"Training: {name}"
        )

        print(
            "  Randomized search..."
        )

        (
            model,
            best_params,
            tuning_score
        ) = tune_classifier(
            name,
            config,
            X,
            y_classification
        )

        print(
            "  Cross-validation..."
        )

        cv_metrics = (
            classification_cross_validation(
                model,
                X,
                y_classification
            )
        )

        train_metrics = (
            evaluate_classifier(
                model,
                X,
                y_classification
            )
        )

        combined = {

            **cv_metrics,

            **train_metrics,

            "best_params":
                best_params,

            "tuning_score":
                (
                    float(tuning_score)
                    if tuning_score is not None
                    else None
                )

        }

        classification_results[
            name
        ] = combined

        print(
            f"  CV Accuracy: "
            f"{cv_metrics['cv_accuracy']:.4f}"
        )

        print(
            f"  CV Macro F1: "
            f"{cv_metrics['cv_macro_f1']:.4f}"
        )

        print(
            f"  CV Balanced Accuracy: "
            f"{cv_metrics['cv_balanced_accuracy']:.4f}"
        )

        print(
            f"  Best Parameters: "
            f"{best_params}"
        )

        score = cv_metrics[
            "cv_macro_f1"
        ]

        if score > best_classifier_score:

            best_classifier_score = (
                score
            )

            best_classifier = (
                model
            )

            best_classifier_name = (
                name
            )

    # ========================================================
    # REGRESSION
    # ========================================================

    print()
    print("-" * 70)
    print("REGRESSION MODEL TUNING")
    print("-" * 70)

    regression_models = (
        get_regression_models()
    )

    regression_results = {}

    best_regressor = None

    best_regressor_name = None

    best_regressor_score = np.inf

    for name, config in (
        regression_models.items()
    ):

        print()
        print(
            f"Training: {name}"
        )

        print(
            "  Randomized search..."
        )

        (
            model,
            best_params,
            tuning_score
        ) = tune_regressor(
            name,
            config,
            X,
            y_regression
        )

        print(
            "  Cross-validation..."
        )

        cv_metrics = (
            regression_cross_validation(
                model,
                X,
                y_regression
            )
        )

        train_metrics = (
            evaluate_regressor(
                model,
                X,
                y_regression
            )
        )

        combined = {

            **cv_metrics,

            **train_metrics,

            "best_params":
                best_params,

            "tuning_score":
                (
                    float(tuning_score)
                    if tuning_score is not None
                    else None
                )

        }

        regression_results[
            name
        ] = combined

        print(
            f"  CV MAE: "
            f"{cv_metrics['cv_mae']:.4f}"
        )

        print(
            f"  CV RMSE: "
            f"{cv_metrics['cv_rmse']:.4f}"
        )

        print(
            f"  CV R2: "
            f"{cv_metrics['cv_r2']:.4f}"
        )

        print(
            f"  Best Parameters: "
            f"{best_params}"
        )

        score = cv_metrics[
            "cv_rmse"
        ]

        if score < best_regressor_score:

            best_regressor_score = (
                score
            )

            best_regressor = (
                model
            )

            best_regressor_name = (
                name
            )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = (
        get_feature_importance(
            best_classifier,
            FEATURE_NAMES
        )
    )

    feature_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # ========================================================
    # SAVE BEST CLASSIFIER
    # ========================================================

    classifier_path = os.path.join(
        MODELS_DIR,
        "academic_performance_model.pkl"
    )

    joblib.dump(
        best_classifier,
        classifier_path
    )

    # ========================================================
    # SAVE BEST REGRESSOR
    # ========================================================

    regressor_path = os.path.join(
        MODELS_DIR,
        "academic_gpa_model.pkl"
    )

    joblib.dump(
        best_regressor,
        regressor_path
    )

    # ========================================================
    # SAVE FEATURE NAMES
    # ========================================================

    feature_path = os.path.join(
        MODELS_DIR,
        "feature_names.pkl"
    )

    joblib.dump(
        FEATURE_NAMES,
        feature_path
    )

    # ========================================================
    # BEST CLASSIFIER METRICS
    # ========================================================

    best_classifier_metrics = (
        classification_results[
            best_classifier_name
        ]
    )

    # ========================================================
    # SAVE EVALUATION JSON
    # ========================================================

    evaluation_results = {

        "dataset": {

            "rows": int(
                len(df)
            ),

            "columns": int(
                len(df.columns)
            ),

            "features":
                FEATURE_NAMES,

            "classification_target":
                CLASSIFICATION_TARGET,

            "regression_target":
                REGRESSION_TARGET

        },

        "data_leakage_protection": {

            "student_id_excluded":
                True,

            "gpa_excluded_from_gradeclass":
                True,

            "gradeclass_excluded_from_features":
                True

        },

        "best_classification_model":
            best_classifier_name,

        "best_regression_model":
            best_regressor_name,

        "classification_models":
            classification_results,

        "regression_models":
            regression_results,

        "feature_importance":
            feature_importance,

        "confusion_matrix":
            best_classifier_metrics[
                "confusion_matrix"
            ],

        "class_labels":
            best_classifier_metrics[
                "class_labels"
            ]

    }

    evaluation_path = os.path.join(
        MODELS_DIR,
        "model_evaluation_results.json"
    )

    with open(
        evaluation_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_results,
            file,
            indent=4
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print()

    print(
        f"Best Classifier: "
        f"{best_classifier_name}"
    )

    print(
        f"Best Classifier CV Accuracy: "
        f"{classification_results[best_classifier_name]['cv_accuracy']:.4f}"
    )

    print(
        f"Best Classifier CV Macro F1: "
        f"{classification_results[best_classifier_name]['cv_macro_f1']:.4f}"
    )

    print(
        f"Best Classifier CV Balanced Accuracy: "
        f"{classification_results[best_classifier_name]['cv_balanced_accuracy']:.4f}"
    )

    print()

    print(
        f"Best Regressor: "
        f"{best_regressor_name}"
    )

    print(
        f"Best Regressor CV RMSE: "
        f"{regression_results[best_regressor_name]['cv_rmse']:.4f}"
    )

    print(
        f"Best Regressor CV R2: "
        f"{regression_results[best_regressor_name]['cv_r2']:.4f}"
    )

    print()

    print(
        "Saved classifier:"
    )

    print(
        classifier_path
    )

    print()

    print(
        "Saved regressor:"
    )

    print(
        regressor_path
    )

    print()

    print(
        "Saved evaluation results:"
    )

    print(
        evaluation_path
    )

    print()

    print("=" * 70)

    return evaluation_results


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        train_all_models()

    except Exception as error:

        print()
        print("=" * 70)
        print("TRAINING ERROR")
        print("=" * 70)

        print(
            str(error)
        )

        print("=" * 70)
        print()

        raise