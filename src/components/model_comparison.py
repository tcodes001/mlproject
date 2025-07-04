import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
import seaborn as sns

def plot_model_comparison(model_report):
    """
    Plots horizontal bar charts comparing R², RMSE, and MAE for all models.
    Saves the figure to artifacts/model_comparison.png
    """
    save_path = "artifacts/model_comparison.png"

    models = list(model_report.keys())
    r2_scores = [model_report[model]["R2"] for model in models]
    rmse_scores = [model_report[model]["RMSE"] for model in models]
    mae_scores = [model_report[model]["MAE"] for model in models]

    #def get_color_palette(n, cmap_name):
     #   cmap = cm.get_cmap(cmap_name)
      #  return [cmap(i / n) for i in range(n)]

    #r2_colors = get_color_palette(len(models), "Purples")
    #rmse_colors = get_color_palette(len(models), "Greens")
    #mae_colors = get_color_palette(len(models), "Blues")
    colors = sns.color_palette("Set2")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    axs[0].barh(models, r2_scores, color=colors, edgecolor='black')
    axs[0].set_title("R² Score")
    axs[0].set_xlabel("R²")

    axs[1].barh(models, rmse_scores, color=colors, edgecolor='black')
    axs[1].set_title("RMSE")
    axs[1].set_xlabel("RMSE")

    axs[2].barh(models, mae_scores, color=colors, edgecolor='black')
    axs[2].set_title("MAE")
    axs[2].set_xlabel("MAE")

    for ax in axs:
        ax.invert_yaxis()  # Highest value at top
        ax.set_xlim(left=0)  # Always start from 0
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print("📊 Model comparison graph saved at", save_path)
