import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

def bias_and_fairness_analysis(model, X_test_df, X_test_arr, y_test):
    
    sensitive_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

    os.makedirs("artifacts", exist_ok=True)
    summaries = []

    y_pred = model.predict(X_test_arr)

    for col in sensitive_columns:
        groups = X_test_df[col].unique()
        r2_scores = []
        mae_scores = []

        for group in groups:
            idx = X_test_df[col] == group
            y_true_group = y_test[idx]
            y_pred_group = y_pred[idx]

            r2 = r2_score(y_true_group, y_pred_group)
            mae = mean_absolute_error(y_true_group, y_pred_group)

            r2_scores.append(r2)
            mae_scores.append(mae)

        # Plot bar chart
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        if col == "gender":
            colors = sns.color_palette("rocket")
            colors2 = sns.color_palette("crest")
        elif col == "race_ethnicity":
            colors = sns.color_palette("dark:salmon_r")
            colors2 = sns.color_palette("YlOrBr")
        elif col == "parental_level_of_education":
            colors = sns.dark_palette("#69d", reverse=True)
            colors2 = sns.light_palette("seagreen")
        elif col == "lunch":
            colors = sns.color_palette("vlag")
            colors2 = sns.color_palette("ch:s=-.2,r=.6")
        else: 
            colors =sns.color_palette("crest")
            colors2 = sns.color_palette("rocket")


        axs[0].bar(groups, r2_scores, color=colors)
        axs[0].set_title(f"{col} - R² Score")
        axs[0].set_ylabel("R²")
        axs[0].tick_params(axis='x', rotation=45)

        axs[1].bar(groups, mae_scores, color=colors2)
        axs[1].set_title(f"{col} - MAE")
        axs[1].set_ylabel("MAE")
        axs[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"artifacts/fairness_{col}.png")
        plt.close()

        # Identify best and worst performing groups based on R²
        if len(groups) >= 2:
        # Find group with highest and lowest R²
            best_idx = np.argmax(r2_scores)
            worst_idx = np.argmin(r2_scores)

            best_group = groups[best_idx]
            worst_group = groups[worst_idx]

            r2_better = r2_scores[best_idx]
            r2_worse = r2_scores[worst_idx]
            mae_better = mae_scores[best_idx]
            mae_worse = mae_scores[worst_idx]

            # Bias strength label based on R² difference
            r2_diff = abs(r2_better - r2_worse)
            bias_strength = "strong" if r2_diff > 0.1 else "small"

            if len(groups) == 2:
                summary = (
                    f"The model performs slightly better for {best_group} "
                    f"(R² = {r2_better:.2f}, MAE = {mae_better:.2f}) compared to {worst_group} "
                    f"(R² = {r2_worse:.2f}, MAE = {mae_worse:.2f}). "
                    f"This suggests a {bias_strength} performance bias favoring {best_group}."
                )
            else:
                summary = (
                    f"The model performs best for {best_group} (R² = {r2_better:.2f}, MAE = {mae_better:.2f}) and worst for {worst_group} "
                    f"(R² = {r2_worse:.2f}, MAE = {mae_worse:.2f}). This suggests a {bias_strength} performance bias favoring {best_group}."
                )

        summaries.append(summary)
        print(f"✅ Bias analysis done for: {col}")

    return summaries
