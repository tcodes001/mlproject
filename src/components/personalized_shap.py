import shap
from collections import defaultdict

def personal_shap(input_df, model, pipeline, train_df):
    # Transform data
    background_data = pipeline.transform(train_df.sample(100, random_state=42))
    transformed_input = pipeline.transform(input_df)

    explainer = shap.Explainer(model.predict, background_data)
    shap_values = explainer(transformed_input)

    base_value = shap_values.base_values[0]
    contributions = shap_values.values[0]

    # Try getting feature names from the pipeline
    try:
        feature_names = pipeline.get_feature_names_out()
    except:
        feature_names = [f'feature_{i}' for i in range(len(contributions))]

    # 🔁 Group SHAP values by their original feature (e.g., combine one-hot encodings)
    grouped_shap = defaultdict(float)

    # Manual mapping: maps substrings in transformed features to clean labels
    mapping = {
        'reading_writing_avg': 'Reading-Writing Avg',
        'gender': 'Gender',
        'race_ethnicity': 'Race Ethnicity',
        'parental_level_of_education': "Parent's Education",
        'lunch': 'Lunch',
        'test_preparation_course': 'Test Preparation Course'
    }

    for name, value in zip(feature_names, contributions):
        # Match any substring key in the feature name
        for key in mapping:
            if key in name:
                grouped_shap[mapping[key]] += value
                break

    # 📦 Format the SHAP explanation nicely
    explainations = []
    for feature, value in grouped_shap.items():
        if abs(value) < 0.5:
            symbol = "🟦"
        elif value > 0:
            symbol = "🟩"
        else:
            symbol = "🟥"

        explainations.append(f" {symbol} {feature} : {value:+.1f} points")

    return base_value, explainations
