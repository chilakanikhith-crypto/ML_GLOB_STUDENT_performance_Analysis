import os
import sys
import json
import pandas as pd

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(
    BASE_DIR,
    "Student_performance_data.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_PATH = os.path.join(
    MODELS_DIR,
    "model_evaluation_results.json"
)

# Add backend directory to Python path
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from ml_models import train_all_models


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# REQUIRED DATASET COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "StudentID",
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
    "Volunteering",
    "GPA",
    "GradeClass"
]


# ============================================================
# ML FEATURES
# ============================================================

ML_FEATURES = [
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


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset():

    print_header("DATASET VALIDATION")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not os.path.exists(CSV_PATH):

        print("ERROR: Dataset not found.")
        print()
        print("Expected location:")
        print(CSV_PATH)

        return None

    print()
    print("Dataset found:")
    print(CSV_PATH)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:

        print()
        print("ERROR: Could not read dataset.")
        print(str(e))

        return None

    print()
    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("ERROR: Missing required columns:")

        for column in missing_columns:
            print(f"  ✗ {column}")

        return None

    print()
    print("All required columns are present.")

    # --------------------------------------------------------
    # Display ML features
    # --------------------------------------------------------

    print()
    print("ML input features:")

    for feature in ML_FEATURES:
        print(f"  ✓ {feature}")

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    print()
    print("Checking for data leakage...")

    if "StudentID" in ML_FEATURES:

        print("  ✗ StudentID is incorrectly included.")
        return None

    print("  ✓ StudentID excluded")

    if "GPA" in ML_FEATURES:

        print("  ✗ GPA is incorrectly included in GradeClass features.")
        return None

    print("  ✓ GPA excluded from GradeClass features")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print()
    print("Missing values:")

    missing_values = df[REQUIRED_COLUMNS].isnull().sum()

    total_missing = int(missing_values.sum())

    if total_missing == 0:

        print("  ✓ No missing values")

    else:

        print("  ✗ Missing values detected:")

        for column, count in missing_values.items():

            if count > 0:
                print(f"    {column}: {count}")

        return None

    # --------------------------------------------------------
    # GradeClass distribution
    # --------------------------------------------------------

    print()
    print("GradeClass distribution:")

    grade_counts = df["GradeClass"].value_counts().sort_index()

    for grade_class, count in grade_counts.items():

        print(
            f"  Class {grade_class}: "
            f"{count} students"
        )

    # --------------------------------------------------------
    # GPA statistics
    # --------------------------------------------------------

    print()
    print("GPA statistics:")

    average_gpa = df["GPA"].mean()
    highest_gpa = df["GPA"].max()
    lowest_gpa = df["GPA"].min()

    print(f"  Average GPA: {average_gpa:.2f}")
    print(f"  Highest GPA: {highest_gpa:.2f}")
    print(f"  Lowest GPA: {lowest_gpa:.2f}")

    print_header("DATASET VALIDATION COMPLETE")

    return df


# ============================================================
# DISPLAY TRAINING RESULTS
# ============================================================

def display_results():

    print_header("FINAL TRAINING SUMMARY")

    if not os.path.exists(RESULTS_PATH):

        print()
        print("ERROR: Evaluation results file not found.")

        return

    try:

        with open(
            RESULTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(file)

    except Exception as e:

        print()
        print("ERROR: Could not read evaluation results.")
        print(str(e))

        return

    # --------------------------------------------------------
    # Best Classification Model
    # --------------------------------------------------------

    best_classifier = results.get(
        "best_classification_model",
        "Not available"
    )

    best_regressor = results.get(
        "best_regression_model",
        "Not available"
    )

    print()
    print("BEST CLASSIFICATION MODEL")
    print("-" * 70)

    if isinstance(best_classifier, dict):

        model_name = (
            best_classifier.get("name")
            or best_classifier.get("model")
            or best_classifier.get("model_name")
            or "Unknown"
        )

        accuracy = best_classifier.get(
            "cv_accuracy",
            best_classifier.get("accuracy", None)
        )

        macro_f1 = best_classifier.get(
            "cv_macro_f1",
            best_classifier.get("macro_f1", None)
        )

        balanced_accuracy = best_classifier.get(
            "cv_balanced_accuracy",
            best_classifier.get("balanced_accuracy", None)
        )

        print(f"Model: {model_name}")

        if accuracy is not None:
            print(f"CV Accuracy: {accuracy:.4f}")

        if macro_f1 is not None:
            print(f"CV Macro F1: {macro_f1:.4f}")

        if balanced_accuracy is not None:
            print(
                f"CV Balanced Accuracy: "
                f"{balanced_accuracy:.4f}"
            )

    else:

        print(f"Model: {best_classifier}")

        # ----------------------------------------------------
        # Try finding details from classification_models
        # ----------------------------------------------------

        classification_models = results.get(
            "classification_models",
            {}
        )

        if isinstance(classification_models, dict):

            if best_classifier in classification_models:

                model_result = classification_models[
                    best_classifier
                ]

                accuracy = model_result.get(
                    "cv_accuracy"
                )

                macro_f1 = model_result.get(
                    "cv_macro_f1"
                )

                balanced_accuracy = model_result.get(
                    "cv_balanced_accuracy"
                )

                if accuracy is not None:
                    print(
                        f"CV Accuracy: "
                        f"{accuracy:.4f}"
                    )

                if macro_f1 is not None:
                    print(
                        f"CV Macro F1: "
                        f"{macro_f1:.4f}"
                    )

                if balanced_accuracy is not None:
                    print(
                        f"CV Balanced Accuracy: "
                        f"{balanced_accuracy:.4f}"
                    )

    # --------------------------------------------------------
    # Best Regression Model
    # --------------------------------------------------------

    print()
    print("BEST REGRESSION MODEL")
    print("-" * 70)

    if isinstance(best_regressor, dict):

        model_name = (
            best_regressor.get("name")
            or best_regressor.get("model")
            or best_regressor.get("model_name")
            or "Unknown"
        )

        rmse = best_regressor.get(
            "cv_rmse",
            best_regressor.get("rmse", None)
        )

        r2 = best_regressor.get(
            "cv_r2",
            best_regressor.get("r2", None)
        )

        print(f"Model: {model_name}")

        if rmse is not None:
            print(f"CV RMSE: {rmse:.4f}")

        if r2 is not None:
            print(f"CV R2: {r2:.4f}")

    else:

        print(f"Model: {best_regressor}")

        # ----------------------------------------------------
        # Try finding details from regression_models
        # ----------------------------------------------------

        regression_models = results.get(
            "regression_models",
            {}
        )

        if isinstance(regression_models, dict):

            if best_regressor in regression_models:

                model_result = regression_models[
                    best_regressor
                ]

                rmse = model_result.get(
                    "cv_rmse"
                )

                r2 = model_result.get(
                    "cv_r2"
                )

                if rmse is not None:
                    print(
                        f"CV RMSE: "
                        f"{rmse:.4f}"
                    )

                if r2 is not None:
                    print(
                        f"CV R2: "
                        f"{r2:.4f}"
                    )

    # --------------------------------------------------------
    # Saved files
    # --------------------------------------------------------

    print()
    print("MODEL FILES")
    print("-" * 70)

    classifier_path = os.path.join(
        MODELS_DIR,
        "academic_performance_model.pkl"
    )

    regressor_path = os.path.join(
        MODELS_DIR,
        "academic_gpa_model.pkl"
    )

    features_path = os.path.join(
        MODELS_DIR,
        "feature_names.pkl"
    )

    results_path = os.path.join(
        MODELS_DIR,
        "model_evaluation_results.json"
    )

    files = [
        ("Classifier", classifier_path),
        ("GPA Regressor", regressor_path),
        ("Feature Names", features_path),
        ("Evaluation Results", results_path)
    ]

    for name, path in files:

        if os.path.exists(path):

            print(f"  ✓ {name}")
            print(f"    {path}")

        else:

            print(f"  ✗ {name}")
            print(f"    Missing: {path}")

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING SUCCESSFUL")
    print("=" * 70)

    print()
    print("The trained models are ready for the Flask application.")

    print()
    print("Next step:")
    print("Start the Flask application and test the dashboard.")

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    df = validate_dataset()

    if df is None:

        print()
        print("Dataset validation failed.")
        sys.exit(1)

    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    print()
    print()
    print("Starting model training...")
    print()

    try:

        train_all_models()

    except KeyboardInterrupt:

        print()
        print("TRAINING INTERRUPTED BY USER.")
        sys.exit(1)

    except Exception as e:

        print()
        print("=" * 70)
        print("TRAINING ERROR")
        print("=" * 70)

        print()
        print(str(e))

        sys.exit(1)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_results()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()