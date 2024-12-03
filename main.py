#from src.pdf_to_csv import extract_transactions_from_pdf
#from src.generate_synData import generate_synthetic_features

# Step 1: Extract data from PDF and save as CSV
#extract_transactions_from_pdf("data/transaction_history_unlocked.pdf", "data/transactions.csv")

# Step 2: Generate synthetic features and save final dataset
#generate_synthetic_features("data/transactions.csv", "data/final_dataset.csv")

# from src.pdf_to_csv import extract_transactions_from_pdf
# from src.generate_synData import generate_synthetic_features

from src.preprocess.preprocess_data import load_data, preprocess_data, split_data
from src.model.train_model import train_model
from src.model.evaluate_model import evaluate_model, tune_hyperparameters
import joblib
import pandas as pd

# Import the new modules for prediction and visualization
from src.predict import make_predictions
from src.visualize_predictions import load_predictions, save_fraudulent_distribution

def main():
    # Step 1: Load and preprocess the dataset
    print("[INFO] Loading the final dataset...")
    data_path = "data/final_dataset.csv"
    df = load_data(data_path)

    print("[INFO] Preprocessing the dataset...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("[INFO] Preprocessing complete. Saving preprocessed data...")
    joblib.dump((X_train, X_test, y_train, y_test), 'src/model/preprocessed_data.joblib')

    # Step 2: Train the model
    print("[INFO] Training the model...")
    rf_model = train_model(X_train, y_train)
    print("[INFO] Model training complete. Saving the model...")

    # Save the trained model
    joblib.dump(rf_model, 'src/model/random_forest_model.joblib')

    # Step 3: Evaluate the model
    print("[INFO] Evaluating the model...")
    evaluate_model(rf_model, X_test, y_test, report_output_path="static/images/classification_report.png")

    # Step 4: Hyperparameter tuning
    print("[INFO] Starting hyperparameter tuning...")
    best_rf_model = tune_hyperparameters(X_train, y_train)
    print("[INFO] Best Hyperparameters:", best_rf_model.get_params())

    # Step 5: Load the best model after hyperparameter tuning (if needed)
    # The best model has already been saved during hyperparameter tuning
    print("[INFO] Loading the best model after tuning...")
    best_rf_model = joblib.load('src/model/best_random_forest_model.joblib')

    # Final evaluation of the best model
    print("[INFO] Evaluating the best model...")
    evaluate_model(best_rf_model, X_test, y_test)

    # Step 6: Make predictions on new data
    print("[INFO] Making predictions on new transaction data...")
    make_predictions()

    # Step 7: Visualize the predictions
    print("[INFO] Visualizing the results of the predictions...")
    predictions_file = 'data/predictions.csv'
    predictions_df = load_predictions(predictions_file)
    output_path = 'static/images/fraudulent_distribution.png'
    save_fraudulent_distribution(predictions_df, output_path)

if __name__ == "__main__":
    main()
