# Import necessary libraries
import joblib
from sklearn.ensemble import RandomForestClassifier

# Step 1: Train the Random Forest model
def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'src/model/random_forest_model.joblib')  # Save the trained model
    return rf_model

# Main execution for testing
if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, y_train, y_test = joblib.load('src/model/preprocessed_data.joblib')
    rf_model = train_model(X_train, y_train)
    print("Model training complete.")
