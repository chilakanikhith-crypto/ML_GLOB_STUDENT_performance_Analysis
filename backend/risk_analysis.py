import pandas as pd

from data_utils import get_feature_columns
from prediction import predict_student


def calculate_risk_level(predicted_gpa, predicted_grade_class, student_data):
    """
    Determine student risk level using ML prediction
    and behavioral indicators.

    This is a screening indicator, not a causal or
    guaranteed academic judgment.
    """

    score = 0
    reasons = []

    study_time = float(student_data.get("StudyTimeWeekly", 0))
    absences = float(student_data.get("Absences", 0))
    tutoring = int(student_data.get("Tutoring", 0))
    parental_support = int(student_data.get("ParentalSupport", 0))

    # -------------------------------------------------
    # Predicted GPA
    # -------------------------------------------------

    if predicted_gpa < 2.0:
        score += 4
        reasons.append("Predicted GPA is below 2.0")

    elif predicted_gpa < 2.5:
        score += 2
        reasons.append("Predicted GPA is below 2.5")

    elif predicted_gpa < 3.0:
        score += 1
        reasons.append("Predicted GPA is below 3.0")

    # -------------------------------------------------
    # Predicted Grade Class
    # -------------------------------------------------

    if predicted_grade_class == 4:
        score += 3
        reasons.append("Predicted Grade Class indicates high academic risk")

    elif predicted_grade_class == 3:
        score += 2
        reasons.append("Predicted Grade Class indicates improvement is needed")

    # -------------------------------------------------
    # Study Time
    # -------------------------------------------------

    if study_time < 5:
        score += 2
        reasons.append("Low weekly study time")

    elif study_time < 10:
        score += 1
        reasons.append("Study time could be increased")

    # -------------------------------------------------
    # Absences
    # -------------------------------------------------

    if absences >= 15:
        score += 3
        reasons.append("High number of absences")

    elif absences >= 10:
        score += 2
        reasons.append("Attendance needs attention")

    elif absences >= 5:
        score += 1
        reasons.append("Some attendance improvement may help")

    # -------------------------------------------------
    # Tutoring
    # -------------------------------------------------

    if tutoring == 0 and predicted_gpa < 2.5:
        score += 1
        reasons.append("Academic tutoring/support may be useful")

    # -------------------------------------------------
    # Parental Support
    # -------------------------------------------------

    if parental_support == 0:
        score += 1
        reasons.append("Additional academic support may be beneficial")

    # -------------------------------------------------
    # Final Risk Level
    # -------------------------------------------------

    if score >= 7:
        risk_level = "High"

    elif score >= 4:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return risk_level, score, reasons


def analyze_student_risk(row):
    """
    Analyze one student using the trained ML models.
    """

    feature_columns = get_feature_columns()

    student_data = {}

    for column in feature_columns:
        value = row[column]

        if pd.isna(value):
            value = 0

        student_data[column] = value

    # -------------------------------------------------
    # ML prediction
    # -------------------------------------------------

    prediction = predict_student(student_data)

    predicted_gpa = float(prediction["predicted_gpa"])
    predicted_grade_class = int(prediction["predicted_grade_class"])

    # -------------------------------------------------
    # Risk calculation
    # -------------------------------------------------

    risk_level, risk_score, reasons = calculate_risk_level(
        predicted_gpa,
        predicted_grade_class,
        student_data
    )

    return {
        "student_id": int(row["StudentID"]),
        "predicted_gpa": round(predicted_gpa, 2),
        "predicted_grade_class": predicted_grade_class,
        "performance": prediction["performance"],
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasons": reasons,
        "study_time": float(row["StudyTimeWeekly"]),
        "absences": int(row["Absences"]),
        "tutoring": int(row["Tutoring"]),
        "parental_support": int(row["ParentalSupport"])
    }


def analyze_all_students(df):
    """
    Analyze every student in the dataset.
    """

    results = []

    for _, row in df.iterrows():

        try:
            result = analyze_student_risk(row)
            results.append(result)

        except Exception as error:
            print(
                f"Risk analysis failed for StudentID "
                f"{row.get('StudentID', 'Unknown')}: {error}"
            )

    return results


def get_risk_summary(results):
    """
    Generate summary statistics from risk results.
    """

    total = len(results)

    high = sum(
        1 for student in results
        if student["risk_level"] == "High"
    )

    medium = sum(
        1 for student in results
        if student["risk_level"] == "Medium"
    )

    low = sum(
        1 for student in results
        if student["risk_level"] == "Low"
    )

    high_percentage = (
        round((high / total) * 100, 2)
        if total > 0
        else 0
    )

    medium_percentage = (
        round((medium / total) * 100, 2)
        if total > 0
        else 0
    )

    low_percentage = (
        round((low / total) * 100, 2)
        if total > 0
        else 0
    )

    return {
        "total_students": total,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "high_percentage": high_percentage,
        "medium_percentage": medium_percentage,
        "low_percentage": low_percentage
    }


if __name__ == "__main__":

    from data_utils import load_data

    print("=" * 70)
    print("AT-RISK STUDENT DETECTION")
    print("=" * 70)

    try:

        df = load_data()

        print(f"\nDataset loaded: {len(df)} students")

        results = analyze_all_students(df)

        summary = get_risk_summary(results)

        print("\n" + "=" * 70)
        print("RISK SUMMARY")
        print("=" * 70)

        print(
            f"Total Students : "
            f"{summary['total_students']}"
        )

        print(
            f"High Risk      : "
            f"{summary['high_risk']} "
            f"({summary['high_percentage']}%)"
        )

        print(
            f"Medium Risk    : "
            f"{summary['medium_risk']} "
            f"({summary['medium_percentage']}%)"
        )

        print(
            f"Low Risk       : "
            f"{summary['low_risk']} "
            f"({summary['low_percentage']}%)"
        )

        print("\n" + "=" * 70)
        print("SAMPLE RESULTS")
        print("=" * 70)

        for student in results[:10]:

            print(
                f"\nStudent ID: {student['student_id']}"
            )

            print(
                f"Predicted GPA: "
                f"{student['predicted_gpa']}"
            )

            print(
                f"Grade Class: "
                f"{student['predicted_grade_class']}"
            )

            print(
                f"Risk Level: "
                f"{student['risk_level']}"
            )

            print(
                f"Risk Score: "
                f"{student['risk_score']}"
            )

            if student["reasons"]:

                print("Reasons:")

                for reason in student["reasons"]:
                    print(f"  - {reason}")

        print("\n" + "=" * 70)
        print("RISK ANALYSIS COMPLETED")
        print("=" * 70)

    except Exception as error:

        print("\nERROR:")
        print(error)