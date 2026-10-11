import os
import pandas as pd

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Student_performance_data_backup.csv"
)


# ============================================================
# MACHINE LEARNING FEATURES
# ============================================================

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
# REQUIRED DATASET COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "StudentID",
    *FEATURE_COLUMNS,
    "GPA",
    "GradeClass"
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_data():
    """Load and validate the student dataset."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError(
            "The student dataset is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "The dataset contains missing values."
        )

    return df


# ============================================================
# GET FEATURE COLUMNS
# ============================================================

def get_feature_columns():
    """Return the approved 12 ML input features."""

    return FEATURE_COLUMNS.copy()


# ============================================================
# GENERATE NEXT STUDENT ID
# ============================================================

def get_next_student_id(df):
    """Generate the next available StudentID."""

    if df.empty or "StudentID" not in df.columns:
        return 1

    student_ids = pd.to_numeric(
        df["StudentID"],
        errors="raise"
    )

    if student_ids.isnull().any():
        raise ValueError(
            "StudentID contains missing values."
        )

    return int(student_ids.max()) + 1


# ============================================================
# VALIDATE GPA
# ============================================================

def validate_gpa(gpa):
    """Validate GPA on the 1-10 scale."""

    try:
        gpa = float(gpa)
    except (TypeError, ValueError):
        raise ValueError(
            "GPA must be a valid number."
        )

    if not 1.0 <= gpa <= 10.0:
        raise ValueError(
            "GPA must be between 1 and 10."
        )

    return gpa


# ============================================================
# CALCULATE GRADE CLASS
# ============================================================

def calculate_grade_class(gpa):
    """
    Convert GPA on the 1-10 scale into GradeClass.

    0 = Excellent
    1 = Very Good
    2 = Good
    3 = Needs Improvement
    4 = At Risk
    """

    gpa = validate_gpa(gpa)

    if gpa >= 8.875:
        return 0

    elif gpa >= 7.75:
        return 1

    elif gpa >= 6.625:
        return 2

    elif gpa >= 5.5:
        return 3

    else:
        return 4


# ============================================================
# GET GRADE LABEL
# ============================================================

def get_grade_label(grade_class):
    """Return the readable label for a GradeClass."""

    labels = {
        0: "Excellent",
        1: "Very Good",
        2: "Good",
        3: "Needs Improvement",
        4: "At Risk"
    }

    try:
        grade_class = int(grade_class)
    except (TypeError, ValueError):
        return "Unknown"

    return labels.get(grade_class, "Unknown")


# ============================================================
# GET PERFORMANCE CATEGORY
# ============================================================

def get_performance_category(gpa):
    """Return the performance category for GPA 1-10."""

    grade_class = calculate_grade_class(gpa)

    return get_grade_label(grade_class)


# ============================================================
# VALIDATE DATASET GPA VALUES
# ============================================================

def validate_dataset_gpa(df):
    """Check whether every dataset GPA is between 1 and 10."""

    if "GPA" not in df.columns:
        raise ValueError(
            "GPA column is missing from the dataset."
        )

    gpa_values = pd.to_numeric(
        df["GPA"],
        errors="coerce"
    )

    if gpa_values.isnull().any():
        raise ValueError(
            "GPA contains invalid or missing values."
        )

    if not gpa_values.between(1, 10).all():
        raise ValueError(
            "All GPA values must be between 1 and 10."
        )

    return True