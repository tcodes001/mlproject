from flask import Flask,request,render_template, send_file, send_from_directory
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.components.personalized_shap import personal_shap
from src.utils import get_background_data
from src.generate_report import generate_model_report
import os


application = Flask(__name__)

app = application

## Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            try:
                data = CustomData(
                    gender = request.form.get('gender'),
                    race_ethnicity = request.form.get ('race_ethnicity'),
                    parental_level_of_education = request.form.get('parental_level_of_education'),
                    lunch = request.form.get('lunch'),
                    test_preparation_course = request.form.get('test_preparation_course'),
                    reading_score = float(request.form.get('reading_score')),
                    writing_score = float(request.form.get('writing_score'))
                )

                pred_df = data.get_data_as_data_frame()
                print(pred_df)
                
                training_input_data = get_background_data()


                predict_pipeline = PredictPipeline()
                results, model, pipeline = predict_pipeline.predict(pred_df)

                print("Prediction results:", results)
                print("Prediction type:", type(results))

                base_value , explainations = personal_shap(pred_df, model, pipeline, training_input_data)

                import re

                # Extract +0.4, -6.3 etc. from explanation strings
                shap_values = []
                for line in explainations:
                    match = re.search(r'([-+]\d+\.\d+)', line)
                    if match:
                        shap_values.append(float(match.group(1)))

                print("DEBUG ---")
                print("Base Value: ", base_value)
                print("Sum of SHAP values: ", sum(shap_values))
                print("Predicted by base + SHAP: ", base_value + sum(shap_values))
                print("Actual Model Prediction: ", results[0])
                return render_template('home.html', results = round(results[0], 2), base_value = round(base_value,2), explainations = explainations)
            
            except Exception as e:
                return "Something went wrong." + str(e)

    except Exception as e: # This will appear in `web.stdout.log`
        return "Something went wrong." + str(e)

@app.route('/model_report')
def model_report():
    try:
        return render_template("final_report.html")
    except Exception as e:
        return "Please download the report instead. Open the HTML locally to see the full version with images."
    
@app.route('/download_report')
def download_report():
    try:
        report_path = os.path.join("templates","final_report.html")
        if not os.path.exists(report_path):
            return "Report file not found. Please run the training pipeline first."

        return send_file(report_path, as_attachment=True, download_name="Student_Model_Report.html")
    except Exception as e:
        return f"Error downloading report: {str(e)}"

@app.route('/artifacts/<path:filename>')
def serve_artifact(filename):
    return send_from_directory('artifacts', filename)
    
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
