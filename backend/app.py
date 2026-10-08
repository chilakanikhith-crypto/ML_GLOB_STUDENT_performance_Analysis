import json
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from data_utils import calculate_grade_class, get_next_student_id, load_data
from ml_models import train_all_models
from prediction import predict_student, what_if_analysis
from risk_analysis import analyze_all_students, get_risk_summary


# ============================================================
# PATH CONFIGURATION
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

TEMPLATES_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

RESULTS_PATH = os.path.join(
    MODELS_DIR,
    "model_evaluation_results.json"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)


# ============================================================
# CONSTANTS
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
# HELPER FUNCTIONS
# ============================================================

def get_dataframe():
    """Load the latest dataset."""
    return load_data()


def clean_value(value):
    """Convert pandas/numpy values into JSON-safe values."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def dataframe_to_records(df):
    """Convert dataframe rows into JSON-safe records."""

    records = []

    for record in df.to_dict(orient="records"):

        records.append({
            key: clean_value(value)
            for key, value in record.items()
        })

    return records


def apply_student_filters(df, params):
    """Apply filters used by the Student Records page."""

    filtered = df.copy()

    # --------------------------------------------------------
    # Student ID
    # --------------------------------------------------------

    student_id = params.get(
        "student_id",
        ""
    ).strip()

    if student_id:

        try:
            filtered = filtered[
                filtered["StudentID"] == int(student_id)
            ]

        except ValueError:
            pass

    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    gender = params.get(
        "gender",
        ""
    ).strip()

    if gender:

        filtered = filtered[
            filtered["Gender"].astype(str) == gender
        ]

    # --------------------------------------------------------
    # Grade Class
    # --------------------------------------------------------

    grade_class = params.get(
        "grade_class",
        ""
    ).strip()

    if grade_class:

        try:
            filtered = filtered[
                filtered["GradeClass"] == float(grade_class)
            ]

        except ValueError:
            pass

    # --------------------------------------------------------
    # Tutoring
    # --------------------------------------------------------

    tutoring = params.get(
        "tutoring",
        ""
    ).strip()

    if tutoring:

        filtered = filtered[
            filtered["Tutoring"].astype(str) == tutoring
        ]

    # --------------------------------------------------------
    # Minimum GPA
    # --------------------------------------------------------

    min_gpa = params.get(
        "min_gpa",
        ""
    ).strip()

    if min_gpa:

        try:
            filtered = filtered[
                filtered["GPA"] >= float(min_gpa)
            ]

        except ValueError:
            pass

    # --------------------------------------------------------
    # Maximum GPA
    # --------------------------------------------------------

    max_gpa = params.get(
        "max_gpa",
        ""
    ).strip()

    if max_gpa:

        try:
            filtered = filtered[
                filtered["GPA"] <= float(max_gpa)
            ]

        except ValueError:
            pass

    # --------------------------------------------------------
    # Maximum Absences
    # --------------------------------------------------------

    max_absences = params.get(
        "max_absences",
        ""
    ).strip()

    if max_absences:

        try:
            filtered = filtered[
                filtered["Absences"] <= float(max_absences)
            ]

        except ValueError:
            pass

    return filtered


def validate_prediction_input(data):
    """
    Validate and convert the 12 ML features.

    StudentID and actual GPA are intentionally excluded.
    """

    if not data:
        return None, "No prediction data received."

    student_data = {}

    for feature in ML_FEATURES:

        if feature not in data:

            return None, (
                f"Missing feature: {feature}"
            )

        try:

            student_data[feature] = float(
                data[feature]
            )

        except (
            ValueError,
            TypeError
        ):

            return None, (
                f"Invalid value for {feature}"
            )

    return student_data, None


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html"
    )


@app.route("/api/dashboard")
def dashboard_api():

    try:

        df = get_dataframe()

        # ----------------------------------------------------
        # KPI statistics
        # ----------------------------------------------------

        total_students = len(df)

        average_gpa = float(
            df["GPA"].mean()
        )

        highest_gpa = float(
            df["GPA"].max()
        )

        lowest_gpa = float(
            df["GPA"].min()
        )

        # ----------------------------------------------------
        # GPA distribution
        # ----------------------------------------------------

        gpa_distribution = []

        bins = [
            0,
            0.5,
            1,
            1.5,
            2,
            2.5,
            3,
            3.5,
            4
        ]

        for i in range(
            len(bins) - 1
        ):

            lower = bins[i]
            upper = bins[i + 1]

            if upper == 4:

                count = int(
                    (
                        (df["GPA"] >= lower)
                        &
                        (df["GPA"] <= upper)
                    ).sum()
                )

            else:

                count = int(
                    (
                        (df["GPA"] >= lower)
                        &
                        (df["GPA"] < upper)
                    ).sum()
                )

            gpa_distribution.append({
                "range":
                    f"{lower:.1f}-{upper:.1f}",

                "count":
                    count
            })

        # ----------------------------------------------------
        # GradeClass distribution
        # ----------------------------------------------------

        grade_distribution = []

        grade_counts = (
            df["GradeClass"]
            .value_counts()
            .sort_index()
        )

        for grade_class, count in (
            grade_counts.items()
        ):

            grade_distribution.append({
                "grade_class":
                    int(grade_class),

                "count":
                    int(count)
            })

        # ----------------------------------------------------
        # Study Time vs GPA
        # ----------------------------------------------------

        study_time_data = []

        for _, row in df[
            [
                "StudyTimeWeekly",
                "GPA"
            ]
        ].iterrows():

            study_time_data.append({
                "study_time":
                    float(row["StudyTimeWeekly"]),

                "gpa":
                    float(row["GPA"])
            })

        # ----------------------------------------------------
        # Absences vs GPA
        # ----------------------------------------------------

        absences_data = []

        for _, row in df[
            [
                "Absences",
                "GPA"
            ]
        ].iterrows():

            absences_data.append({
                "absences":
                    float(row["Absences"]),

                "gpa":
                    float(row["GPA"])
            })

        # ----------------------------------------------------
        # Parental Support vs GPA
        # ----------------------------------------------------

        parental_support_data = []

        support_group = (
            df.groupby(
                "ParentalSupport"
            )["GPA"]
            .mean()
            .sort_index()
        )

        for support, gpa in (
            support_group.items()
        ):

            parental_support_data.append({
                "support":
                    int(support),

                "gpa":
                    float(gpa)
            })

        # ----------------------------------------------------
        # Activity vs GPA
        # ----------------------------------------------------

        activity_data = []

        activity_columns = [
            "Extracurricular",
            "Sports",
            "Music",
            "Volunteering"
        ]

        for column in activity_columns:

            active_rows = df[
                df[column] == 1
            ]

            if len(active_rows) > 0:

                average_gpa = float(
                    active_rows["GPA"].mean()
                )

            else:

                average_gpa = 0.0

            activity_data.append({
                "activity":
                    column,

                "average_gpa":
                    average_gpa
            })

        # ----------------------------------------------------
        # GPA by GradeClass
        # ----------------------------------------------------

        grade_gpa_data = []

        grade_gpa = (
            df.groupby(
                "GradeClass"
            )["GPA"]
            .mean()
            .sort_index()
        )

        for grade_class, gpa in (
            grade_gpa.items()
        ):

            grade_gpa_data.append({
                "grade_class":
                    int(grade_class),

                "average_gpa":
                    float(gpa)
            })

        return jsonify({

            "success":
                True,

            "total_students":
                total_students,

            "average_gpa":
                average_gpa,

            "highest_gpa":
                highest_gpa,

            "lowest_gpa":
                lowest_gpa,

            "gpa_distribution":
                gpa_distribution,

            "grade_distribution":
                grade_distribution,

            "study_time_data":
                study_time_data,

            "absences_data":
                absences_data,

            "parental_support_data":
                parental_support_data,

            "activity_data":
                activity_data,

            "grade_gpa_data":
                grade_gpa_data
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# STUDENTS
# ============================================================

@app.route("/students")
def students():

    return render_template(
        "students.html"
    )


@app.route("/api/students")
def students_api():

    try:

        df = get_dataframe()

        filtered = apply_student_filters(
            df,
            request.args
        )

        return jsonify({

            "success":
                True,

            "total":
                len(filtered),

            "students":
                dataframe_to_records(
                    filtered
                )
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/students/download")
def download_students():

    try:

        df = get_dataframe()

        filtered = apply_student_filters(
            df,
            request.args
        )

        output_path = os.path.join(
            BASE_DIR,
            "filtered_students.csv"
        )

        filtered.to_csv(
            output_path,
            index=False
        )

        return send_file(
            output_path,
            as_attachment=True,
            download_name="filtered_students.csv"
        )

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# ADD STUDENT
# ============================================================

@app.route("/add-student")
def add_student():

    return render_template(
        "add_student.html"
    )


@app.route("/api/next-student-id")
def next_student_id_api():

    try:

        df = load_data()

        next_id = get_next_student_id(
            df
        )

        return jsonify({

            "success":
                True,

            "next_student_id":
                int(next_id)
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route(
    "/api/add-student",
    methods=["POST"]
)
def add_student_api():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error":
                    "No student data received."
            }), 400

        # ----------------------------------------------------
        # Load current dataset
        # ----------------------------------------------------

        df = load_data()

        # ----------------------------------------------------
        # Generate ID
        # ----------------------------------------------------

        student_id = get_next_student_id(
            df
        )

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        required_input_fields = [

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
            "GPA"
        ]

        for field in required_input_fields:

            if field not in data:

                return jsonify({
                    "success": False,
                    "error":
                        f"Missing field: {field}"
                }), 400

        # ----------------------------------------------------
        # Parse values
        # ----------------------------------------------------

        try:

            age = int(
                data["Age"]
            )

            gender = int(
                data["Gender"]
            )

            ethnicity = int(
                data["Ethnicity"]
            )

            parental_education = int(
                data["ParentalEducation"]
            )

            study_time = float(
                data["StudyTimeWeekly"]
            )

            absences = int(
                data["Absences"]
            )

            tutoring = int(
                data["Tutoring"]
            )

            parental_support = int(
                data["ParentalSupport"]
            )

            extracurricular = int(
                data["Extracurricular"]
            )

            sports = int(
                data["Sports"]
            )

            music = int(
                data["Music"]
            )

            volunteering = int(
                data["Volunteering"]
            )

            gpa = float(
                data["GPA"]
            )

        except (
            ValueError,
            TypeError
        ):

            return jsonify({
                "success": False,
                "error":
                    "Invalid student input."
            }), 400

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if age < 1:

            return jsonify({
                "success": False,
                "error":
                    "Age must be positive."
            }), 400

        if study_time < 0:

            return jsonify({
                "success": False,
                "error":
                    "Study time cannot be negative."
            }), 400

        if absences < 0:

            return jsonify({
                "success": False,
                "error":
                    "Absences cannot be negative."
            }), 400

        if gpa < 0 or gpa > 4:

            return jsonify({
                "success": False,
                "error":
                    "GPA must be between 0 and 4."
            }), 400

        # ----------------------------------------------------
        # Calculate GradeClass
        # ----------------------------------------------------

        grade_class = calculate_grade_class(
            gpa
        )

        # ----------------------------------------------------
        # Create record
        # ----------------------------------------------------

        new_student = {

            "StudentID":
                int(student_id),

            "Age":
                age,

            "Gender":
                gender,

            "Ethnicity":
                ethnicity,

            "ParentalEducation":
                parental_education,

            "StudyTimeWeekly":
                study_time,

            "Absences":
                absences,

            "Tutoring":
                tutoring,

            "ParentalSupport":
                parental_support,

            "Extracurricular":
                extracurricular,

            "Sports":
                sports,

            "Music":
                music,

            "Volunteering":
                volunteering,

            "GPA":
                gpa,

            "GradeClass":
                grade_class
        }

        # ----------------------------------------------------
        # Append to CSV
        # ----------------------------------------------------

        new_row = pd.DataFrame(
            [new_student]
        )

        updated_df = pd.concat(
            [
                df,
                new_row
            ],
            ignore_index=True
        )

        updated_df.to_csv(
            CSV_PATH,
            index=False
        )

        return jsonify({

            "success":
                True,

            "message":
                "Student added successfully.",

            "student":
                new_student
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# PREDICTION / AI ACADEMIC ADVISOR
# ============================================================

@app.route("/predict")
def predict():

    return render_template(
        "predict.html"
    )


@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict_api():

    try:

        data = request.get_json()

        student_data, error = (
            validate_prediction_input(data)
        )

        if error:

            return jsonify({
                "success": False,
                "error": error
            }), 400

        result = predict_student(
            student_data
        )

        return jsonify({

            "success":
                True,

            "result":
                result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# MODEL LAB
# ============================================================

@app.route("/model-lab")
def model_lab():

    return render_template(
        "model_lab.html"
    )


@app.route("/api/model-lab")
def model_lab_api():

    try:

        if not os.path.exists(
            RESULTS_PATH
        ):

            return jsonify({
                "success": False,
                "error":
                    "Model results not found."
            }), 404

        with open(
            RESULTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(file)

        return jsonify({

            "success":
                True,

            **results
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

@app.route("/what-if")
def what_if():

    return render_template(
        "what_if.html"
    )


@app.route(
    "/api/what-if",
    methods=["POST"]
)
def what_if_api():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error":
                    "No scenario data received."
            }), 400

        baseline_data = data.get(
            "baseline"
        )

        scenario_data = data.get(
            "scenario"
        )

        if not baseline_data:

            return jsonify({
                "success": False,
                "error":
                    "Baseline data is required."
            }), 400

        if not scenario_data:

            return jsonify({
                "success": False,
                "error":
                    "Scenario data is required."
            }), 400

        for feature in ML_FEATURES:

            if feature not in baseline_data:

                return jsonify({
                    "success": False,
                    "error":
                        f"Baseline missing {feature}"
                }), 400

            if feature not in scenario_data:

                return jsonify({
                    "success": False,
                    "error":
                        f"Scenario missing {feature}"
                }), 400

            try:

                baseline_data[feature] = float(
                    baseline_data[feature]
                )

                scenario_data[feature] = float(
                    scenario_data[feature]
                )

            except (
                ValueError,
                TypeError
            ):

                return jsonify({
                    "success": False,
                    "error":
                        f"Invalid value for {feature}"
                }), 400

        result = what_if_analysis(
            baseline_data,
            scenario_data
        )

        return jsonify({

            "success":
                True,

            "result":
                result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RETRAIN
# ============================================================

@app.route("/retrain")
def retrain():

    return render_template(
        "retrain.html"
    )


@app.route(
    "/api/retrain",
    methods=["POST"]
)
def retrain_api():

    try:

        df = load_data()

        # ----------------------------------------------------
        # Validate columns
        # ----------------------------------------------------

        missing_columns = [

            column

            for column in REQUIRED_COLUMNS

            if column not in df.columns

        ]

        if missing_columns:

            return jsonify({

                "success":
                    False,

                "error":
                    "Missing columns: "
                    +
                    ", ".join(
                        missing_columns
                    )
            }), 400

        # ----------------------------------------------------
        # Minimum records
        # ----------------------------------------------------

        if len(df) < 20:

            return jsonify({
                "success": False,
                "error":
                    "At least 20 records required."
            }), 400

        # ----------------------------------------------------
        # Leakage protection
        # ----------------------------------------------------

        if "StudentID" in ML_FEATURES:

            return jsonify({
                "success": False,
                "error":
                    "StudentID cannot be an ML feature."
            }), 400

        if "GPA" in ML_FEATURES:

            return jsonify({
                "success": False,
                "error":
                    "GPA cannot be used to predict GradeClass."
            }), 400

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        if df[
            REQUIRED_COLUMNS
        ].isnull().any().any():

            return jsonify({
                "success": False,
                "error":
                    "Dataset contains missing values."
            }), 400

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_all_models()

        # ----------------------------------------------------
        # Read results
        # ----------------------------------------------------

        if not os.path.exists(
            RESULTS_PATH
        ):

            return jsonify({
                "success": False,
                "error":
                    "Evaluation results were not saved."
            }), 500

        with open(
            RESULTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(file)

        return jsonify({

            "success":
                True,

            "message":
                "Models retrained successfully.",

            "best_classification_model":
                results.get(
                    "best_classification_model"
                ),

            "best_regression_model":
                results.get(
                    "best_regression_model"
                ),

            "results":
                results
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# AT-RISK STUDENT DETECTION
# ============================================================

@app.route("/at-risk")
def at_risk():

    return render_template(
        "at_risk.html"
    )


@app.route("/api/at-risk")
def at_risk_api():

    try:

        df = get_dataframe()

        if df.empty:

            return jsonify({

                "success":
                    True,

                "summary": {

                    "total_students":
                        0,

                    "high_risk":
                        0,

                    "medium_risk":
                        0,

                    "low_risk":
                        0,

                    "high_percentage":
                        0,

                    "medium_percentage":
                        0,

                    "low_percentage":
                        0
                },

                "students":
                    []
            })

        # ----------------------------------------------------
        # Run ML-based risk analysis
        # ----------------------------------------------------

        results = analyze_all_students(
            df
        )

        # ----------------------------------------------------
        # Generate summary
        # ----------------------------------------------------

        summary = get_risk_summary(
            results
        )

        # ----------------------------------------------------
        # Sort High -> Medium -> Low
        # ----------------------------------------------------

        risk_order = {
            "High": 0,
            "Medium": 1,
            "Low": 2
        }

        results.sort(
            key=lambda student: (

                risk_order.get(
                    student["risk_level"],
                    3
                ),

                -student["risk_score"],

                student["student_id"]
            )
        )

        return jsonify({

            "success":
                True,

            "summary":
                summary,

            "students":
                results
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# STATUS
# ============================================================

@app.route("/api/status")
def status_api():

    try:

        df = load_data()

        classifier_exists = os.path.exists(
            os.path.join(
                MODELS_DIR,
                "academic_performance_model.pkl"
            )
        )

        regressor_exists = os.path.exists(
            os.path.join(
                MODELS_DIR,
                "academic_gpa_model.pkl"
            )
        )

        results_exists = os.path.exists(
            RESULTS_PATH
        )

        return jsonify({

            "success":
                True,

            "dataset_exists":
                os.path.exists(
                    CSV_PATH
                ),

            "students":
                len(df),

            "classifier_ready":
                classifier_exists,

            "regressor_ready":
                regressor_exists,

            "evaluation_ready":
                results_exists,

            "features":
                len(ML_FEATURES),

            "leakage_protection":
                True
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "API endpoint not found."
        }), 404

    return """
    <h1>404 - Page Not Found</h1>
    <p>The requested page does not exist.</p>
    """, 404


@app.errorhandler(500)
def internal_server_error(error):

    if request.path.startswith(
        "/api/"
    ):

        return jsonify({

            "success":
                False,

            "error":
                "Internal server error."
        }), 500

    return """
    <h1>500 - Internal Server Error</h1>
    <p>Please check the Flask terminal.</p>
    """, 500


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ACADEMIC PERFORMANCE ANALYTICS")
    print("=" * 60)

    print()
    print("Dataset:", CSV_PATH)

    try:

        df = load_data()

        print(
            "Students loaded:",
            len(df)
        )

    except Exception as e:

        print(
            "Warning:",
            str(e)
        )

    print()
    print("Available pages:")

    print(
        "Dashboard      : "
        "http://127.0.0.1:5000/"
    )

    print(
        "Students       : "
        "http://127.0.0.1:5000/students"
    )

    print(
        "Add Student    : "
        "http://127.0.0.1:5000/add-student"
    )

    print(
        "Predict        : "
        "http://127.0.0.1:5000/predict"
    )

    print(
        "Model Lab      : "
        "http://127.0.0.1:5000/model-lab"
    )

    print(
        "What-If        : "
        "http://127.0.0.1:5000/what-if"
    )

    print(
        "Retrain        : "
        "http://127.0.0.1:5000/retrain"
    )

    print(
        "At-Risk        : "
        "http://127.0.0.1:5000/at-risk"
    )

    print("=" * 60)

    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000)),
    debug=False
)