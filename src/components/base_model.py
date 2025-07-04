from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_dummy_baseline(train_arr, test_arr):
    
    X_train = train_arr[:, :-1]
    y_train = train_arr[:, -1]
    X_test = test_arr[:, :-1]
    y_test = test_arr[:, -1]

    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)

    baseline_preds = dummy.predict(X_test)

    baseline_r2 = r2_score(y_test, baseline_preds)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

    print(f"📉 Dummy Baseline Model Performance:")
    print(f"  R² Score: {baseline_r2:.4f}")
    print(f"  RMSE: {baseline_rmse:.4f}")
    print(f"  MAE: {baseline_mae:.4f}")

    return {
        "R2": baseline_r2,
        "RMSE": baseline_rmse,
        "MAE": baseline_mae
    }
