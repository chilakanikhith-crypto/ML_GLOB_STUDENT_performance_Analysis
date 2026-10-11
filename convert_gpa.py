import pandas as pd
import shutil
from pathlib import Path

# Project dataset
csv_path = Path("Student_performance_data.csv")
backup_path = Path("Student_performance_data_backup.csv")

# Check dataset
if not csv_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {csv_path.resolve()}"
    )

# Read dataset
df = pd.read_csv(csv_path)

if "GPA" not in df.columns:
    raise ValueError("GPA column not found in dataset.")

# Preserve the original dataset
if not backup_path.exists():
    shutil.copy2(csv_path, backup_path)

# Convert GPA from 0–4 scale to 1–10 scale
old_gpa = pd.to_numeric(df["GPA"], errors="raise")

if old_gpa.isna().any() or not old_gpa.between(0, 4).all():
    raise ValueError("GPA values must be between 0 and 4.")

df["GPA"] = 1 + (old_gpa / 4) * 9

# Update GradeClass using converted GPA thresholds
def calculate_grade_class(gpa):
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

df["GradeClass"] = df["GPA"].apply(
    calculate_grade_class
)

# Save converted dataset
df.to_csv(csv_path, index=False)

print("GPA conversion completed!")
print(f"Backup saved at: {backup_path.resolve()}")
print(f"Minimum GPA: {df['GPA'].min():.2f}")
print(f"Maximum GPA: {df['GPA'].max():.2f}")
print(f"Average GPA: {df['GPA'].mean():.2f}")
print("\nGradeClass distribution:")
print(df["GradeClass"].value_counts().sort_index())