import os
import glob
import base64
from jinja2 import Environment, FileSystemLoader

def image_to_base64(image_path):
    """Convert image file to base64 string for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            base64_string = base64.b64encode(img_data).decode('utf-8')
            
            # Determine image format based on file extension
            if image_path.lower().endswith('.png'):
                return f"data:image/png;base64,{base64_string}"
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                return f"data:image/jpeg;base64,{base64_string}"
            else:
                # Default to PNG if unknown
                return f"data:image/png;base64,{base64_string}"
    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}")
        return ""
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def generate_model_report(model_name, r2_score, rmse, mae, fairness_summaries, confidence_intervals, coverage_percent, baseline_metrics):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report_template.html')

    model_comp_path = "artifacts/model_comparison.png"
    model_comp_img = image_to_base64(model_comp_path)

    sensitive_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

    fairness_imgs_path = [f"artifacts/fairness_{col}.png" for col in sensitive_columns
                      if os.path.exists(f"artifacts/fairness_{col}.png")]

    # Convert all fairness images to base64
    fairness_data = []
    for img_path, summary in zip(fairness_imgs_path, fairness_summaries):
        img_base64 = image_to_base64(img_path)
        fairness_data.append((img_base64, summary))
    

    rendered_html = template.render(
        model_name=model_name,
        r2_score=f"{r2_score:.4f}",
        rmse=f"{rmse:.4f}",
        mae=f"{mae:.4f}",
        baseline_r2=f"{baseline_metrics['R2']:.4f}",
        baseline_rmse=f"{baseline_metrics['RMSE']:.4f}",
        baseline_mae=f"{baseline_metrics['MAE']:.4f}",
        confidence_intervals=confidence_intervals,
        coverage_percent=coverage_percent,
        model_comp_img=model_comp_img,              
        fairness_data=fairness_data
    )

    output_path = os.path.join("templates", "final_report.html")
    with open(output_path, "w") as f:
        f.write(rendered_html)

    return output_path
