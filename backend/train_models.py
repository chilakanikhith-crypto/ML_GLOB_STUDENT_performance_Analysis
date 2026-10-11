import os
import sys
import json
import pandas as pd

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "Student_performance_data.csv"
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

RESULTS_PATH = os.path.join(
    MODELS_DIR,
    "model_evaluation_results.json"
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from ml_models import train_all_models


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

FEATURE_COLUMNS = [
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
# DISPLAY HELPER
# ============================================================

def print_header(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset():
    print_header("ACADEMIC PERFORMANCE ANALYTICS")
    print("Checking dataset...")

    if not os.path.exists(CSV_PATH):
        print("\nERROR: Dataset not found.")
        print("Expected path:", CSV_PATH)
        return None

    try:
        df = pd.read_csv(CSV_PATH)

    except Exception as error:
        print("\nERROR: Could not read CSV.")
        print(error)
        return None

    print("Dataset path:", CSV_PATH)
    print("Dataset shape:", df.shape)

    if df.empty:
        print("ERROR: Dataset is empty.")
        return None

    # Check required columns.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Missing columns:")

        for column in missing_columns:
            print("-", column)

        return None

    # Check missing values.
    if df[REQUIRED_COLUMNS].isnull().any().any():
        print("\nERROR: Missing values detected.")
        print(df[REQUIRED_COLUMNS].isnull().sum())
        return None

    # Convert required columns to numeric values.
    for column in REQUIRED_COLUMNS:
        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.isna().any():
            print(
                f"\nERROR: Invalid numeric values "
                f"found in {column}."
            )
            return None

        df[column] = converted

    # Check infinite values.
    if not all(
        df[REQUIRED_COLUMNS]
        .map(lambda value: abs(float(value)) != float("inf"))
        .all()
    ):
        print("\nERROR: Infinite values detected.")
        return None

    # Check duplicate student IDs.
    if df["StudentID"].duplicated().any():
        print("\nERROR: Duplicate StudentID values found.")
        return None

    # Check student IDs.
    if (df["StudentID"] <= 0).any():
        print("\nERROR: StudentID must be positive.")
        return None

    if not (df["StudentID"] % 1 == 0).all():
        print("\nERROR: StudentID must contain integers.")
        return None

    # Check GPA range.
    if not df["GPA"].between(1, 10).all():
        print("\nERROR: GPA must be between 1 and 10.")
        print("Check the CSV conversion before training.")
        return None

    # Check GradeClass values.
    valid_classes = {0, 1, 2, 3, 4}
    actual_classes = set(df["GradeClass"].unique())

    if not actual_classes.issubset(valid_classes):
        print(
            "\nERROR: GradeClass must contain only "
            "values 0, 1, 2, 3, 4."
        )
        return None

    if not (df["GradeClass"] % 1 == 0).all():
        print("\nERROR: GradeClass values must be integers.")
        return None

    if df["GradeClass"].nunique() < 2:
        print(
            "\nERROR: At least two GradeClass categories "
            "are required."
        )
        return None

    # Verify the approved model features.
    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            print("\nERROR: Missing feature:", feature)
            return None

    # Check that at least two examples exist in each class
    # for stratified training and validation.
    class_counts = df["GradeClass"].value_counts()

    if class_counts.min() < 2:
        print(
            "\nERROR: Every GradeClass represented in the "
            "dataset needs at least two students."
        )
        print(class_counts.sort_index())
        return None

    # Display validation summary.
    print("\nDATASET VALIDATION SUCCESSFUL")
    print("-" * 65)

    print("GPA scale: 1-10")
    print("Number of input features:", len(FEATURE_COLUMNS))
    print("StudentID excluded from model features.")
    print("Actual GPA excluded from classification inputs.")

    print("\nGPA Statistics")
    print("-" * 30)
    print(f"Average GPA: {df['GPA'].mean():.2f}")
    print(f"Lowest GPA: {df['GPA'].min():.2f}")
    print(f"Highest GPA: {df['GPA'].max():.2f}")

    print("\nGradeClass Distribution")
    print("-" * 30)
    print(
        df["GradeClass"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# DISPLAY TRAINING RESULTS
# ============================================================

def display_results():
    print_header("MODEL TRAINING RESULTS")

    if not os.path.exists(RESULTS_PATH):
        print("ERROR: Evaluation results file not found.")
        print("Expected path:", RESULTS_PATH)
        return False

    try:
        with open(
            RESULTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            results = json.load(file)

    except Exception as error:
        print("ERROR: Could not read evaluation results.")
        print(error)
        return False

    # Classification results.
    print("\nCLASSIFICATION RESULTS")
    print("-" * 65)

    print(
        "Best classifier:",
        results.get(
            "best_classifier",
            results.get("best_classification_model", "Unavailable")
        )
    )

    print(
        "Best Macro F1:",
        results.get(
            "best_classifier_macro_f1",
            "Unavailable"
        )
    )

    classification_models = results.get(
        "classification_models",
        {}
    )

    for name, metrics in classification_models.items():
        print(f"\n{name}")
        print("-" * 30)
        print("Accuracy:", metrics.get("accuracy", "N/A"))
        print("Precision:", metrics.get("precision", "N/A"))
        print("Recall:", metrics.get("recall", "N/A"))

        print(
            "Macro F1:",
            metrics.get(
                "f1_score",
                metrics.get("macro_f1", "N/A")
            )
        )

    # Regression results.
    print("\nGPA REGRESSION RESULTS")
    print("-" * 65)

    print(
        "Best regressor:",
        results.get(
            "best_regressor",
            results.get("best_regression_model", "Unavailable")
        )
    )

    print(
        "Best MAE:",
        results.get("best_regressor_mae", "Unavailable")
    )

    regression_models = results.get(
        "regression_models",
        {}
    )

    for name, metrics in regression_models.items():
        print(f"\n{name}")
        print("-" * 30)
        print("MAE:", metrics.get("mae", "N/A"))
        print("RMSE:", metrics.get("rmse", "N/A"))
        print("R2:", metrics.get("r2", "N/A"))

    # Check model files.
    print("\nSAVED MODEL FILES")
    print("-" * 65)

    expected_files = [
        "academic_performance_model.pkl",
        "academic_gpa_model.pkl",
        "feature_names.pkl",
        "model_evaluation_results.json"
    ]

    all_exist = True

    for filename in expected_files:
        path = os.path.join(MODELS_DIR, filename)

        if os.path.exists(path):
            print("[OK]", filename)
        else:
            print("[MISSING]", filename)
            all_exist = False

    if all_exist:
        print_header("TRAINING COMPLETED SUCCESSFULLY")
        return True

    print("\nSome model files are missing.")
    return False


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    # Step 1: Validate dataset.
    df = validate_dataset()

    if df is None:
        sys.exit(1)

    # Step 2: Create models directory.
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Step 3: Train the models.
    print_header("STARTING MODEL TRAINING")

    try:
        results = train_all_models()

        if not isinstance(results, dict):
            print(
                "ERROR: train_all_models() did not "
                "return a dictionary."
            )
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

    except Exception as error:
        print_header("TRAINING FAILED")
        print("Error type:", type(error).__name__)
        print("Error message:", str(error))
        sys.exit(1)

    # Step 4: Display results.
    if not display_results():
        sys.exit(1)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()