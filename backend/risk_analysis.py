from data_utils import (
    get_feature_columns,
    get_grade_label
)

from prediction import predict_student


def analyze_student(student):
    """Calculate a model-based academic risk assessment."""

    features = get_feature_columns()

    student_data = {
        feature: float(student[feature])
        for feature in features
    }

    prediction = predict_student(student_data)

    predicted_gpa = float(
        prediction["predicted_gpa"]
    )

    predicted_class = int(
        prediction["predicted_grade_class"]
    )

    reasons = []

    # Determine initial risk level from model predictions.
    if predicted_gpa < 5.5 or predicted_class == 4:
        risk_level = "High"
        risk_score = 85

    elif predicted_gpa < 6.625 or predicted_class == 3:
        risk_level = "Medium"
        risk_score = 55

    else:
        risk_level = "Low"
        risk_score = 20

    # Additional screening signals.
    study_time = float(student["StudyTimeWeekly"])
    absences = float(student["Absences"])
    tutoring = int(student["Tutoring"])
    parental_support = int(student["ParentalSupport"])

    if study_time < 5:
        reasons.append("Low weekly study time")
        risk_score += 5

    if absences >= 10:
        reasons.append("High number of absences")
        risk_score += 5

    if tutoring == 0:
        reasons.append("No tutoring reported")

    if parental_support <= 1:
        reasons.append("Low parental support reported")

    if predicted_gpa < 5.5:
        reasons.append("Predicted GPA is below 5.5")

    if predicted_class == 4:
        reasons.append("Predicted GradeClass is At Risk")

    risk_score = min(100, max(0, risk_score))

    return {
        "student_id": int(student["StudentID"]),
        "predicted_gpa": round(predicted_gpa, 2),
        "predicted_grade_class": predicted_class,
        "performance_category": get_grade_label(
            predicted_class
        ),
        "risk_level": risk_level,
        "risk_score": int(risk_score),
        "reasons": reasons,
        "gpa_scale": "1-10"
    }


def analyze_all_students(df):
    """Assess all students in the dataset."""

    results = []

    for _, student in df.iterrows():
        try:
            results.append(
                analyze_student(student)
            )

        except Exception as exc:
            results.append({
                "student_id": int(student["StudentID"]),
                "risk_level": "Unknown",
                "risk_score": 0,
                "reasons": [
                    f"Unable to analyze student: {exc}"
                ]
            })

    return results


def get_risk_summary(results):
    """Summarize the risk levels across students."""

    total = len(results)

    high = sum(
        item["risk_level"] == "High"
        for item in results
    )

    medium = sum(
        item["risk_level"] == "Medium"
        for item in results
    )

    low = sum(
        item["risk_level"] == "Low"
        for item in results
    )

    unknown = sum(
        item["risk_level"] == "Unknown"
        for item in results
    )

    def percentage(count):
        return round(
            (count / total) * 100, 2
        ) if total else 0

    return {
        "total_students": total,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "unknown_risk": unknown,
        "high_percentage": percentage(high),
        "medium_percentage": percentage(medium),
        "low_percentage": percentage(low),
        "unknown_percentage": percentage(unknown)
    }