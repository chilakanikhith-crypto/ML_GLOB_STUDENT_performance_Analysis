import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "Student_performance_data.csv")


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


def load_data():
    """Load the student performance dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)


def get_feature_columns():
    """Return the ML feature columns."""
    return FEATURE_COLUMNS.copy()


def get_next_student_id(df):
    """Generate the next StudentID."""
    if df.empty or "StudentID" not in df.columns:
        return 1

    return int(df["StudentID"].max()) + 1


def calculate_grade_class(gpa):
    """
    Convert GPA into GradeClass.

    0 = GPA >= 3.5
    1 = GPA >= 3.0
    2 = GPA >= 2.5
    3 = GPA >= 2.0
    4 = GPA < 2.0
    """

    gpa = float(gpa)

    if gpa >= 3.5:
        return 0
    elif gpa >= 3.0:
        return 1
    elif gpa >= 2.5:
        return 2
    elif gpa >= 2.0:
        return 3
    else:
        return 4


def get_grade_label(grade_class):
    """Return a readable label for GradeClass."""

    labels = {
        0: "Excellent",
        1: "Very Good",
        2: "Good",
        3: "Needs Improvement",
        4: "At Risk"
    }

    return labels.get(int(grade_class), "Unknown")