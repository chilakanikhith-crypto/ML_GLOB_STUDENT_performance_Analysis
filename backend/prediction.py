import os
import joblib
import pandas as pd

from data_utils import (
    calculate_grade_class,
    get_grade_label,
    get_feature_columns
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASSIFIER_PATH = os.path.join(
    MODELS_DIR,
    "academic_performance_model.pkl"
)

REGRESSOR_PATH = os.path.join(
    MODELS_DIR,
    "academic_gpa_model.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    MODELS_DIR,
    "feature_names.pkl"
)


def load_prediction_models():
    """Load the trained classifier, regressor, and features."""

    for path in [
        CLASSIFIER_PATH,
        REGRESSOR_PATH,
        FEATURE_NAMES_PATH
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model file not found: {path}. "
                "Train the models before making predictions."
            )

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    expected_features = get_feature_columns()

    if list(feature_names) != expected_features:
        raise ValueError(
            "Saved model feature names do not match "
            "the 12 approved features. Retrain the models."
        )

    return classifier, regressor, feature_names


def prepare_input(student_data, feature_names):
    """Validate and prepare the student's 12 input features."""

    if not isinstance(student_data, dict):
        raise ValueError(
            "Student data must be provided as a dictionary."
        )

    missing = [
        feature
        for feature in feature_names
        if feature not in student_data
    ]

    if missing:
        raise ValueError(
            f"Missing prediction features: {missing}"
        )

    values = {}

    for feature in feature_names:
        try:
            value = float(student_data[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be a valid number."
            )

        if not pd.notna(value):
            raise ValueError(
                f"{feature} cannot be empty or NaN."
            )

        values[feature] = value

    input_df = pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names
    )

    return input_df


def predict_student(student_data):
    """
    Predict GPA and GradeClass using the 12 approved features.

    StudentID is an identifier, not a model feature.
    Actual GPA is never used as a prediction input.
    """

    classifier, regressor, feature_names = (
        load_prediction_models()
    )

    input_df = prepare_input(
        student_data,
        feature_names
    )

    # Predict GPA using the regression model.
    raw_gpa = float(
        regressor.predict(input_df)[0]
    )

    predicted_gpa = max(
        1.0,
        min(10.0, raw_gpa)
    )

    # Predict GradeClass independently using the classifier.
    predicted_class = int(
        classifier.predict(input_df)[0]
    )

    result = {
        "predicted_gpa": round(predicted_gpa, 2),
        "predicted_grade_class": predicted_class,
        "performance_category": get_grade_label(
            predicted_class
        ),
        "description": (
            "Prediction based on the trained model and "
            "the student's demographic and study features."
        ),
        "gpa_scale": "1-10"
    }

    # Include class probabilities if supported by the model.
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(
            input_df
        )[0]

        result["class_probabilities"] = {
            str(int(class_id)): round(
                float(probability), 4
            )
            for class_id, probability in zip(
                classifier.classes_,
                probabilities
            )
        }

    return result


def what_if_analysis(baseline_data, scenario_data):
    """
    Compare predictions for baseline and changed student features.

    Returns predicted GPA changes and whether the predicted
    GradeClass changed. This comparison does not establish causation.
    """

    baseline = predict_student(baseline_data)
    scenario = predict_student(scenario_data)

    baseline_gpa = baseline["predicted_gpa"]
    scenario_gpa = scenario["predicted_gpa"]

    return {
        "baseline": baseline,
        "scenario": scenario,
        "gpa_change": round(
            scenario_gpa - baseline_gpa,
            2
        ),
        "grade_class_changed": (
            baseline["predicted_grade_class"]
            != scenario["predicted_grade_class"]
        ),
        "explanation": (
            "This is a model-based comparison, not proof "
            "that changing a factor will cause the result."
        )
    }