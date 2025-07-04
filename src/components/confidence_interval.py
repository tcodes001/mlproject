import numpy as np

def compute_prediction_intervals(model, X_test_arr, y_test, X_test_raw_df):

    n_bootstrap=1000
    confidence_level=0.95
    rng = np.random.default_rng()
    n_samples = min(5, X_test_arr.shape[0])
    summaries = []

    for i in range(n_samples):
        predictions = []

        for _ in range(n_bootstrap):
            indices = rng.integers(0, X_test_arr.shape[0], X_test_arr.shape[0])
            X_resample = X_test_arr[indices]
            y_resample = y_test[indices]

            model_clone = type(model)()
            model_clone.fit(X_resample, y_resample)
            pred = model_clone.predict(X_test_arr[i].reshape(1, -1))[0]
            predictions.append(pred)

        lower_bound = np.percentile(predictions, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(predictions, (1 + confidence_level) / 2 * 100)
        point_pred = model.predict(X_test_arr[i].reshape(1, -1))[0]

        # Extract raw info (like gender, lunch, etc.)
        raw_row = X_test_raw_df.iloc[i].to_dict()
        raw_info = ", ".join([f"{k} = {v}" for k, v in raw_row.items()])

        summary = (
            f"Sample {i+1} ➤ {raw_info} ➤ Predicted score is {point_pred:.2f}, "
            f"with a {int(confidence_level * 100)}% confidence interval of "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]."
        )

        print(summary)
        summaries.append(summary)


    full_predictions = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, X_test_arr.shape[0], X_test_arr.shape[0])
        X_resample = X_test_arr[indices]
        y_resample = y_test[indices]

        model_clone = type(model)()
        model_clone.fit(X_resample, y_resample)
        preds = model_clone.predict(X_test_arr)
        full_predictions.append(preds)

    full_predictions = np.array(full_predictions)
    lower = np.percentile(full_predictions, (1 - confidence_level) / 2 * 100, axis=0)
    upper = np.percentile(full_predictions, (1 + confidence_level) / 2 * 100, axis=0)

    y_test = np.array(y_test)
    coverage_count = np.sum((y_test >= lower) & (y_test <= upper))
    coverage_percent = round(coverage_count / len(y_test) * 100, 2)

    return summaries, coverage_percent
