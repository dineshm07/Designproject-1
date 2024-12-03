import pandas as pd
import joblib

# Load the preprocessor and the trained model
preprocessor = joblib.load('src/model/preprocessor.joblib')  # This is where the preprocessor was saved
best_rf_model = joblib.load('src/model/best_random_forest_model.joblib')  # The trained model

def make_predictions():
    # Load new transaction data (assuming it's a CSV file)
    new_data = pd.read_csv('data/new_transaction_data.csv')

    # Print the original shape of the data (for debugging)
    print(f"Original new data shape: {new_data.shape}")

    # Drop non-feature columns (columns not used in model training)
    X_new = new_data.drop(columns=["Date", "Transaction_Time"])  # Adjust if more columns need to be dropped

    # Print the shape of the data after dropping the non-feature columns
    print(f"New data shape after dropping non-feature columns: {X_new.shape}")

    # Preprocess the new data using the preprocessor pipeline (transform, not fit_transform)
    X_new_processed = preprocessor.transform(X_new)

    # Print the shape of the processed data (for debugging)
    print(f"Processed new data shape: {X_new_processed.shape}")

    # Check if the number of features matches what the model expects
    if X_new_processed.shape[1] != best_rf_model.n_features_in_:
        print(f"Feature mismatch! Expected {best_rf_model.n_features_in_} features, but got {X_new_processed.shape[1]}.")
        return

    # Make predictions on the processed data
    y_pred = best_rf_model.predict(X_new_processed)

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(y_pred, columns=["Fraudulent_Prediction"])
    predictions_df.to_csv("data/predictions.csv", index=False)

    print("[INFO] Predictions saved to 'data/predictions.csv'.")

if __name__ == "__main__":
    make_predictions()
