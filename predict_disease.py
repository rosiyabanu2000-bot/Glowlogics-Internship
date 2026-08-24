import joblib
import pandas as pd


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

data = joblib.load(
    "best_disease_model.pkl"
)

model = data["model"]

label_encoder = data["label_encoder"]

features = data["features"]


# =====================================================
# TITLE
# =====================================================

print("=" * 60)
print("        DISEASE PREDICTION SYSTEM")
print("=" * 60)


# =====================================================
# DISPLAY SYMPTOMS
# =====================================================

print("\nAvailable Symptoms:\n")

for i, symptom in enumerate(
    features,
    start=1
):

    print(
        f"{i}. {symptom}"
    )


# =====================================================
# USER INPUT
# =====================================================

print("\n")

user_input = input(
    "Enter your symptoms separated by commas:\n"
)


selected_symptoms = [

    symptom.strip()

    for symptom in user_input.split(",")

]


# =====================================================
# CREATE INPUT DATAFRAME
# =====================================================

input_data = pd.DataFrame(
    0,
    index=[0],
    columns=features
)


# =====================================================
# SET SELECTED SYMPTOMS = 1
# =====================================================

for symptom in selected_symptoms:

    if symptom in input_data.columns:

        input_data.loc[
            0,
            symptom
        ] = 1

    else:

        print(
            f"\nWarning: {symptom} "
            "is not found in the dataset."
        )


# =====================================================
# PREDICTION
# =====================================================

prediction = model.predict(
    input_data
)


predicted_disease = label_encoder.inverse_transform(
    prediction
)[0]


# =====================================================
# RESULT
# =====================================================

print("\n" + "=" * 60)

print(
    "Predicted Disease:",
    predicted_disease
)

print("=" * 60)


print(
    "\nDisclaimer:"
)

print(
    "This prediction is for educational/"
    "internship purposes only and is not "
    "a medical diagnosis."
)