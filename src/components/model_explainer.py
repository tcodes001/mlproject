import shap
import matplotlib.pyplot as plt
import os
import sys
from src.exception import CustomException
import logging

def generate_shap_plot(model, X_train):
    try:
        logging.info("Generating SHAP summary plot...")

        # Define the output path for the SHAP plot
        output_path = "artifacts/shap_summary_plot.png"

        # Initialize SHAP explainer
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        # Generate and save the SHAP summary plot
        shap.summary_plot(shap_values, X_train, show=False)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        logging.info(f"SHAP plot saved to {output_path}")

    except Exception as e:
        logging.error("Exception occurred while generating SHAP plot.")
        raise CustomException(e, sys)
