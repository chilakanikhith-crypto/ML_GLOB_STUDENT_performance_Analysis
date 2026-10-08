import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


CLASSIFICATION_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "academic_performance_model.pkl"
)


REGRESSION_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "academic_gpa_model.pkl"
)


FEATURE_NAMES_PATH = os.path.join(
    MODELS_DIR,
    "feature_names.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not os.path.exists(
        CLASSIFICATION_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Classification model not found. "
            "Please train the models first."
        )


    if not os.path.exists(
        REGRESSION_MODEL_PATH
    ):

        raise FileNotFoundError(
            "GPA regression model not found. "
            "Please train the models first."
        )


    if not os.path.exists(
        FEATURE_NAMES_PATH
    ):

        raise FileNotFoundError(
            "Feature names file not found. "
            "Please train the models first."
        )


    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )


    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )


    feature_names = joblib.load(
        FEATURE_NAMES_PATH
    )


    return (
        classification_model,
        regression_model,
        feature_names
    )


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_student_input(
    student_data,
    feature_names
):

    missing_features = [

        feature

        for feature in feature_names

        if feature not in student_data

    ]


    if missing_features:

        raise ValueError(

            "Missing required features: "
            + ", ".join(
                missing_features
            )

        )


    # --------------------------------------------------------
    # Validate numeric values
    # --------------------------------------------------------

    for feature in feature_names:

        try:

            float(
                student_data[feature]
            )

        except (
            ValueError,
            TypeError
        ):

            raise ValueError(

                f"{feature} must be numeric."

            )


    # --------------------------------------------------------
    # GPA is NOT checked here.
    #
    # Actual GPA is deliberately not required
    # for prediction.
    # --------------------------------------------------------

    return True


# ============================================================
# CREATE FEATURE DATAFRAME
# ============================================================

def prepare_features(
    student_data,
    feature_names
):

    values = []

    for feature in feature_names:

        values.append(
            float(
                student_data[feature]
            )
        )


    X = pd.DataFrame(

        [values],

        columns=feature_names

    )


    return X


# ============================================================
# GRADE CLASS DESCRIPTION
# ============================================================

def grade_class_description(
    grade_class
):

    descriptions = {

        0:
            "Excellent Performance",

        1:
            "Very Good Performance",

        2:
            "Good / Average Performance",

        3:
            "Needs Improvement",

        4:
            "At Risk / Needs Significant Improvement"

    }


    return descriptions.get(

        int(grade_class),

        "Unknown Performance Category"

    )


# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

def performance_category(
    grade_class
):

    categories = {

        0:
            "Excellent",

        1:
            "Very Good",

        2:
            "Average",

        3:
            "Needs Improvement",

        4:
            "At Risk"

    }


    return categories.get(

        int(grade_class),

        "Unknown"

    )


# ============================================================
# PREDICT STUDENT
# ============================================================

def predict_student(
    student_data
):

    (
        classification_model,
        regression_model,
        feature_names
    ) = load_models()


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_student_input(
        student_data,
        feature_names
    )


    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = prepare_features(

        student_data,

        feature_names

    )


    # ========================================================
    # GPA PREDICTION
    # ========================================================

    predicted_gpa = regression_model.predict(
        X
    )[0]


    predicted_gpa = float(
        np.clip(
            predicted_gpa,
            0,
            4
        )
    )


    # ========================================================
    # GRADE CLASS PREDICTION
    # ========================================================

    predicted_class = classification_model.predict(
        X
    )[0]


    predicted_class = int(
        predicted_class
    )


    # ========================================================
    # PERFORMANCE CATEGORY
    # ========================================================

    category = performance_category(
        predicted_class
    )


    description = grade_class_description(
        predicted_class
    )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    probabilities = None


    if hasattr(
        classification_model,
        "predict_proba"
    ):

        probability_values = (
            classification_model
            .predict_proba(X)[0]
        )


        classes = (
            classification_model
            .classes_
        )


        probabilities = {

            str(int(class_value)):
                round(
                    float(probability),
                    4
                )

            for class_value, probability
            in zip(
                classes,
                probability_values
            )

        }


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "predicted_gpa":
            round(
                predicted_gpa,
                2
            ),

        "predicted_grade_class":
            predicted_class,

        "performance_category":
            category,

        "description":
            description,

        "class_probabilities":
            probabilities

    }


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

def what_if_analysis(
    baseline_data,
    scenario_data
):

    baseline_result = predict_student(
        baseline_data
    )


    scenario_result = predict_student(
        scenario_data
    )


    # --------------------------------------------------------
    # GPA difference
    # --------------------------------------------------------

    gpa_change = (

        scenario_result[
            "predicted_gpa"
        ]

        -

        baseline_result[
            "predicted_gpa"
        ]

    )


    # --------------------------------------------------------
    # Grade class difference
    # --------------------------------------------------------

    class_change = (

        scenario_result[
            "predicted_grade_class"
        ]

        -

        baseline_result[
            "predicted_grade_class"
        ]

    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This is model-based scenario analysis.
    # It does NOT establish causation.
    # --------------------------------------------------------

    return {

        "baseline":
            baseline_result,

        "scenario":
            scenario_result,

        "changes": {

            "predicted_gpa_change":
                round(
                    float(gpa_change),
                    2
                ),

            "grade_class_change":
                int(
                    class_change
                )

        },

        "note":
            "What-If results represent "
            "model-based scenario estimates "
            "and should not be interpreted "
            "as causal effects."

    }


# ============================================================
# TEST MODULE DIRECTLY
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "ACADEMIC PERFORMANCE ANALYTICS"
    )
    print(
        "PREDICTION ENGINE TEST"
    )
    print("=" * 60)
    print()


    try:

        (
            classification_model,
            regression_model,
            feature_names
        ) = load_models()


        print(
            "Models loaded successfully."
        )

        print()

        print(
            "Features:"
        )

        for feature in feature_names:

            print(
                f"  - {feature}"
            )


        # ----------------------------------------------------
        # Example student
        # ----------------------------------------------------

        example_student = {

            "Age": 18,

            "Gender": 1,

            "Ethnicity": 0,

            "ParentalEducation": 2,

            "StudyTimeWeekly": 15,

            "Absences": 5,

            "Tutoring": 1,

            "ParentalSupport": 3,

            "Extracurricular": 1,

            "Sports": 1,

            "Music": 0,

            "Volunteering": 1

        }


        print()

        print(
            "Running example prediction..."
        )


        result = predict_student(
            example_student
        )


        print()

        print(
            "Predicted GPA:",
            result[
                "predicted_gpa"
            ]
        )


        print(
            "Predicted Grade Class:",
            result[
                "predicted_grade_class"
            ]
        )


        print(
            "Performance Category:",
            result[
                "performance_category"
            ]
        )


        if (
            result[
                "class_probabilities"
            ]
            is not None
        ):

            print()

            print(
                "Class Probabilities:"
            )

            for (
                grade_class,
                probability
            ) in result[
                "class_probabilities"
            ].items():

                print(

                    f"  Class {grade_class}: "
                    f"{probability * 100:.2f}%"

                )


        print()

        print(
            "=" * 60
        )

        print(
            "TEST COMPLETE"
        )

        print(
            "=" * 60
        )


    except Exception as error:

        print()

        print(
            "ERROR:"
        )

        print(
            str(error)
        )