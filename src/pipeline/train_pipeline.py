from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.components.model_comparison import plot_model_comparison
from src.components.fair_analysis import bias_and_fairness_analysis
from src.components.confidence_interval import compute_prediction_intervals
from src.generate_report import generate_model_report
from src.components.base_model import evaluate_dummy_baseline

if __name__ == "__main__":
    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation() 
    train_array, test_array, _ , X_test_raw_df,X_test_arr, y_test_raw_df= data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    _, model_report,best_model, best_model_name, best_model_metrics = model_trainer.initiate_model_trainer(train_array, test_array)

    #Step 4: Model Comparison Graphs
    plot_model_comparison(model_report)

    #Step 5: Bias and Fairness analysis
    bias_summaries = bias_and_fairness_analysis(model=best_model, X_test_df=X_test_raw_df, X_test_arr= X_test_arr, y_test=y_test_raw_df)
    
    #Step 6: Confidence Interval
    ci_summaries, coverage_percent = compute_prediction_intervals(model=best_model,X_test_arr=X_test_arr, y_test=y_test_raw_df.to_numpy(), X_test_raw_df=X_test_raw_df)

    # Step 7: Baseline Model
    baseline_scores = evaluate_dummy_baseline(train_arr=train_array, test_arr=test_array) 

    #Step 8: Report Generation
    html_content = generate_model_report(model_name=best_model_name, 
                                         r2_score=best_model_metrics["R2"], 
                                         rmse =best_model_metrics["RMSE"],
                                         mae= best_model_metrics["MAE"],
                                         fairness_summaries= bias_summaries,
                                         confidence_intervals= ci_summaries,
                                         coverage_percent= coverage_percent,
                                         baseline_metrics = baseline_scores)
     