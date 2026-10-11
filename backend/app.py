import os
import json
import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file
)

from data_utils import (
    calculate_grade_class,
    get_next_student_id,
    load_data
)

from ml_models import train_all_models
from prediction import predict_student, what_if_analysis
from risk_analysis import analyze_all_students, get_risk_summary


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

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


# ============================================================
# DATASET COLUMNS AND FEATURES
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

GRADE_LABELS = {
    0: "Excellent",
    1: "Very Good",
    2: "Good",
    3: "Needs Improvement",
    4: "At Risk"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_dataframe():
    """Load the current student dataset."""
    return load_data()


def clean_value(value):
    """Convert pandas and numpy values into JSON-safe values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value


def dataframe_to_records(df):
    """Convert dataframe rows into JSON-safe dictionaries."""
    return [
        {
            key: clean_value(value)
            for key, value in record.items()
        }
        for record in df.to_dict(orient="records")
    ]


def validate_dataset_columns(df):
    """Check that all required columns exist."""
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    return True


def apply_student_filters(df, params):
    """Filter student records using request parameters."""
    filtered = df.copy()

    student_id = params.get("student_id", "").strip()
    if student_id:
        try:
            filtered = filtered[
                filtered["StudentID"] == int(student_id)
            ]
        except ValueError:
            pass

    gender = params.get("gender", "").strip()
    if gender:
        filtered = filtered[
            filtered["Gender"].astype(str) == gender
        ]

    grade_class = params.get("grade_class", "").strip()
    if grade_class:
        try:
            filtered = filtered[
                filtered["GradeClass"] == int(grade_class)
            ]
        except ValueError:
            pass

    tutoring = params.get("tutoring", "").strip()
    if tutoring:
        filtered = filtered[
            filtered["Tutoring"].astype(str) == tutoring
        ]

    min_gpa = params.get("min_gpa", "").strip()
    if min_gpa:
        try:
            filtered = filtered[
                filtered["GPA"] >= float(min_gpa)
            ]
        except ValueError:
            pass

    max_gpa = params.get("max_gpa", "").strip()
    if max_gpa:
        try:
            filtered = filtered[
                filtered["GPA"] <= float(max_gpa)
            ]
        except ValueError:
            pass

    max_absences = params.get("max_absences", "").strip()
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
    Validate the 12 input features.
    Actual GPA and StudentID are not prediction inputs.
    """
    if not isinstance(data, dict):
        return None, "Please provide valid prediction data."

    student_data = {}

    for feature in ML_FEATURES:
        if feature not in data:
            return None, f"Missing feature: {feature}"

        try:
            value = float(data[feature])
        except (ValueError, TypeError):
            return None, f"Invalid value for {feature}."

        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            return None, f"Invalid numeric value for {feature}."

        student_data[feature] = value

    return student_data, None


def read_model_results():
    """Read model evaluation results from JSON."""
    if not os.path.exists(RESULTS_PATH):
        return {}

    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def json_error(error, status_code=500):
    """Return a consistent JSON error response."""
    return jsonify({
        "success": False,
        "error": str(error)
    }), status_code


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/dashboard")
def dashboard_api():
    try:
        df = get_dataframe()
        validate_dataset_columns(df)

        total_students = int(len(df))
        average_gpa = float(df["GPA"].mean())
        highest_gpa = float(df["GPA"].max())
        lowest_gpa = float(df["GPA"].min())

        # GPA distribution for the 1-10 scale.
        bins = list(range(1, 11))
        gpa_distribution = []

        for index in range(len(bins) - 1):
            lower = bins[index]
            upper = bins[index + 1]

            if upper == 10:
                count = int(
                    (
                        (df["GPA"] >= lower)
                        & (df["GPA"] <= upper)
                    ).sum()
                )
            else:
                count = int(
                    (
                        (df["GPA"] >= lower)
                        & (df["GPA"] < upper)
                    ).sum()
                )

            gpa_distribution.append({
                "range": f"{lower}-{upper}",
                "count": count
            })

        # GradeClass distribution.
        grade_distribution = [
            {
                "grade_class": int(grade),
                "label": GRADE_LABELS.get(int(grade), "Unknown"),
                "count": int(count)
            }
            for grade, count in (
                df["GradeClass"].value_counts().sort_index().items()
            )
        ]

        # Study time vs GPA.
        study_time_data = [
            {
                "study_time": float(row["StudyTimeWeekly"]),
                "gpa": float(row["GPA"])
            }
            for _, row in df[
                ["StudyTimeWeekly", "GPA"]
            ].iterrows()
        ]

        # Absences vs GPA.
        absences_data = [
            {
                "absences": float(row["Absences"]),
                "gpa": float(row["GPA"])
            }
            for _, row in df[
                ["Absences", "GPA"]
            ].iterrows()
        ]

        # Parental support vs GPA.
        support_group = (
            df.groupby("ParentalSupport")["GPA"]
            .mean()
            .sort_index()
        )

        parental_support_data = [
            {
                "support": int(support),
                "gpa": float(gpa)
            }
            for support, gpa in support_group.items()
        ]

        # Activities vs GPA.
        activity_data = []

        for column in [
            "Extracurricular",
            "Sports",
            "Music",
            "Volunteering"
        ]:
            active_rows = df[df[column] == 1]

            activity_data.append({
                "activity": column,
                "average_gpa": (
                    float(active_rows["GPA"].mean())
                    if not active_rows.empty
                    else 0.0
                )
            })

        # GPA by GradeClass.
        grade_gpa = df.groupby("GradeClass")["GPA"].mean()

        grade_gpa_data = [
            {
                "grade_class": int(grade),
                "average_gpa": float(gpa)
            }
            for grade, gpa in grade_gpa.items()
        ]

        return jsonify({
            "success": True,
            "gpa_scale": "1-10",
            "total_students": total_students,
            "average_gpa": average_gpa,
            "highest_gpa": highest_gpa,
            "lowest_gpa": lowest_gpa,
            "gpa_distribution": gpa_distribution,
            "grade_distribution": grade_distribution,
            "study_time_data": study_time_data,
            "absences_data": absences_data,
            "parental_support_data": parental_support_data,
            "activity_data": activity_data,
            "grade_gpa_data": grade_gpa_data
        })

    except Exception as error:
        return json_error(error)


# ============================================================
# STUDENT RECORDS
# ============================================================

@app.route("/students")
def students():
    return render_template("students.html")


@app.route("/api/students")
def students_api():
    try:
        df = get_dataframe()
        filtered = apply_student_filters(df, request.args)

        return jsonify({
            "success": True,
            "total": int(len(filtered)),
            "students": dataframe_to_records(filtered)
        })

    except Exception as error:
        return json_error(error)


@app.route("/api/students/download")
def download_students():
    try:
        df = get_dataframe()
        filtered = apply_student_filters(df, request.args)

        output_path = os.path.join(
            BASE_DIR,
            "filtered_students.csv"
        )

        filtered.to_csv(output_path, index=False)

        return send_file(
            output_path,
            as_attachment=True,
            download_name="filtered_students.csv"
        )

    except Exception as error:
        return json_error(error)


# ============================================================
# ADD STUDENT
# ============================================================

@app.route("/add-student")
def add_student():
    return render_template("add_student.html")


@app.route("/api/next-student-id")
def next_student_id_api():
    try:
        df = load_data()

        return jsonify({
            "success": True,
            "next_student_id": int(get_next_student_id(df))
        })

    except Exception as error:
        return json_error(error)


@app.route("/api/add-student", methods=["POST"])
def add_student_api():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return json_error("No valid student data received.", 400)

        df = load_data()
        validate_dataset_columns(df)

        required_fields = ML_FEATURES + ["GPA"]

        for field in required_fields:
            if field not in data:
                return json_error(f"Missing field: {field}", 400)

        try:
            student = {}

            for field in ML_FEATURES:
                student[field] = float(data[field])

            gpa = float(data["GPA"])

        except (ValueError, TypeError):
            return json_error("Invalid student input.", 400)

        if not pd.Series([gpa]).between(1, 10).all():
            return json_error("GPA must be between 1 and 10.", 400)

        if student["Age"] < 1:
            return json_error("Age must be positive.", 400)

        if student["StudyTimeWeekly"] < 0:
            return json_error("Study time cannot be negative.", 400)

        if student["Absences"] < 0:
            return json_error("Absences cannot be negative.", 400)

        # Validate binary activity fields.
        for field in [
            "Gender",
            "Tutoring",
            "Extracurricular",
            "Sports",
            "Music",
            "Volunteering"
        ]:
            if student[field] not in (0, 1):
                return json_error(
                    f"{field} must be either 0 or 1.",
                    400
                )

        # Save new record.
        student_id = get_next_student_id(df)
        grade_class = calculate_grade_class(gpa)

        new_student = {
            "StudentID": int(student_id),
            **student,
            "GPA": gpa,
            "GradeClass": int(grade_class)
        }

        updated_df = pd.concat(
            [df, pd.DataFrame([new_student])],
            ignore_index=True
        )

        updated_df.to_csv(CSV_PATH, index=False)

        return jsonify({
            "success": True,
            "message": "Student added successfully.",
            "student": new_student
        })

    except Exception as error:
        return json_error(error)


# ============================================================
# PREDICTION / AI ACADEMIC ADVISOR
# ============================================================

@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(silent=True)

        student_data, error = validate_prediction_input(data)

        if error:
            return json_error(error, 400)

        result = predict_student(student_data)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:
        return json_error(error)


# ============================================================
# MODEL LAB
# ============================================================

@app.route("/model-lab")
def model_lab():
    return render_template("model_lab.html")


@app.route("/api/model-lab")
def model_lab_api():
    try:
        results = read_model_results()

        if not results:
            return json_error(
                "Model evaluation results not found. "
                "Train the models first.",
                404
            )

        return jsonify({
            "success": True,
            **results
        })

    except Exception as error:
        return json_error(error)


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

@app.route("/what-if")
def what_if():
    return render_template("what_if.html")


@app.route("/api/what-if", methods=["POST"])
def what_if_api():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return json_error("No scenario data received.", 400)

        baseline_data = data.get("baseline")
        scenario_data = data.get("scenario")

        if not isinstance(baseline_data, dict):
            return json_error("Baseline data is required.", 400)

        if not isinstance(scenario_data, dict):
            return json_error("Scenario data is required.", 400)

        baseline, error = validate_prediction_input(baseline_data)

        if error:
            return json_error(f"Baseline: {error}", 400)

        scenario, error = validate_prediction_input(scenario_data)

        if error:
            return json_error(f"Scenario: {error}", 400)

        result = what_if_analysis(baseline, scenario)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as error:
        return json_error(error)


# ============================================================
# RETRAIN MODELS
# ============================================================

@app.route("/retrain")
def retrain():
    return render_template("retrain.html")


@app.route("/api/retrain", methods=["POST"])
def retrain_api():
    try:
        df = load_data()
        validate_dataset_columns(df)

        if len(df) < 20:
            return json_error(
                "At least 20 student records are required.",
                400
            )

        if "StudentID" in ML_FEATURES or "GPA" in ML_FEATURES:
            return json_error(
                "Invalid feature configuration: data leakage detected.",
                400
            )

        if df[REQUIRED_COLUMNS].isnull().any().any():
            return json_error(
                "Dataset contains missing values.",
                400
            )

        if not df["GPA"].between(1, 10).all():
            return json_error(
                "GPA must be on the 1-10 scale.",
                400
            )

        results = train_all_models()

        if not isinstance(results, dict):
            results = read_model_results()

        if not results:
            return json_error(
                "Model evaluation results were not saved.",
                500
            )

        return jsonify({
            "success": True,
            "message": "Models retrained successfully.",
            "best_classification_model": results.get(
                "best_classifier",
                results.get("best_classification_model")
            ),
            "best_regression_model": results.get(
                "best_regressor",
                results.get("best_regression_model")
            ),
            "results": results
        })

    except Exception as error:
        return json_error(error)


# ============================================================
# AT-RISK STUDENT DETECTION
# ============================================================

@app.route("/at-risk")
def at_risk():
    return render_template("at_risk.html")


@app.route("/api/at-risk")
def at_risk_api():
    try:
        df = get_dataframe()

        if df.empty:
            return jsonify({
                "success": True,
                "summary": {
                    "total_students": 0,
                    "high_risk": 0,
                    "medium_risk": 0,
                    "low_risk": 0,
                    "high_percentage": 0,
                    "medium_percentage": 0,
                    "low_percentage": 0
                },
                "students": []
            })

        results = analyze_all_students(df)
        summary = get_risk_summary(results)

        risk_order = {
            "High": 0,
            "Medium": 1,
            "Low": 2
        }

        results.sort(
            key=lambda student: (
                risk_order.get(student.get("risk_level"), 3),
                -student.get("risk_score", 0),
                student.get("student_id", 0)
            )
        )

        return jsonify({
            "success": True,
            "summary": summary,
            "students": results
        })

    except Exception as error:
        return json_error(error)


# ============================================================
# APPLICATION STATUS
# ============================================================

@app.route("/api/status")
def status_api():
    try:
        df = load_data()

        classifier_path = os.path.join(
            MODELS_DIR,
            "academic_performance_model.pkl"
        )

        regressor_path = os.path.join(
            MODELS_DIR,
            "academic_gpa_model.pkl"
        )

        return jsonify({
            "success": True,
            "dataset_exists": os.path.exists(CSV_PATH),
            "students": int(len(df)),
            "classifier_ready": os.path.exists(classifier_path),
            "regressor_ready": os.path.exists(regressor_path),
            "evaluation_ready": os.path.exists(RESULTS_PATH),
            "features": len(ML_FEATURES),
            "gpa_scale": "1-10",
            "leakage_protection": True
        })

    except Exception as error:
        return json_error(error)


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "API endpoint not found."
        }), 404

    return (
        "<h1>404 - Page Not Found</h1>"
        "<p>The requested page does not exist.</p>",
        404
    )


@app.errorhandler(500)
def internal_server_error(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "Internal server error. Check the Flask terminal."
        }), 500

    return (
        "<h1>500 - Internal Server Error</h1>"
        "<p>Please check the Flask terminal.</p>",
        500
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ACADEMIC PERFORMANCE ANALYTICS")
    print("=" * 60)

    print("\nDataset:", CSV_PATH)

    try:
        df = load_data()
        print("Students loaded:", len(df))

    except Exception as error:
        print("Warning:", error)

    print("\nAvailable pages:")
    print("Dashboard   : http://127.0.0.1:5000/")
    print("Students    : http://127.0.0.1:5000/students")
    print("Add Student : http://127.0.0.1:5000/add-student")
    print("Predict     : http://127.0.0.1:5000/predict")
    print("Model Lab   : http://127.0.0.1:5000/model-lab")
    print("What-If     : http://127.0.0.1:5000/what-if")
    print("Retrain     : http://127.0.0.1:5000/retrain")
    print("At-Risk     : http://127.0.0.1:5000/at-risk")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )