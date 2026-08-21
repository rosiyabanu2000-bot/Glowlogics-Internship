import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

study_hours = np.round(np.random.uniform(1, 8, n), 1)
attendance = np.round(np.random.uniform(50, 100, n), 1)
previous_score = np.round(np.random.uniform(40, 95, n), 1)
assignments_completed = np.random.randint(3, 11, n)
sleep_hours = np.round(np.random.uniform(5, 9, n), 1)
extracurricular = np.random.choice(["Yes", "No"], n)

final_score = (
    study_hours * 4
    + attendance * 0.25
    + previous_score * 0.35
    + assignments_completed * 1.5
    + sleep_hours * 1.5
    + np.where(extracurricular == "Yes", 3, 0)
    + np.random.normal(0, 5, n)
)

final_score = np.clip(final_score, 0, 100)
final_score = np.round(final_score, 1)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_score": previous_score,
    "assignments_completed": assignments_completed,
    "sleep_hours": sleep_hours,
    "extracurricular": extracurricular,
    "final_score": final_score
})

df.to_csv("dataset.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print(df.head())