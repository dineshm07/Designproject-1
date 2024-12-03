from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, report_output_path="static/images/classification_report.png"):
    """
    Evaluate the model using the test dataset.

    Parameters:
    - model: Trained classifier model
    - X_test: Features for testing
    - y_test: True labels for testing
    - report_output_path: Path to save the classification report visualization

    Returns:
    None
    """
    print("[INFO] Evaluating the model...")

    # Predict the labels for the test set
    y_pred = model.predict(X_test)

    # Generate the classification report
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save the classification report as a text file
    report_txt_path = report_output_path.replace(".png", ".txt")
    with open(report_txt_path, "w") as file:
        file.write(report)
    print(f"[INFO] Classification report saved as text at {report_txt_path}")

    # Visualize the classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().round(2)
    df_report = df_report.drop("support", axis=1)  # Drop 'support' column for cleaner visuals

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Classification Report")
    plt.savefig(report_output_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Classification report visualization saved as PNG at {report_output_path}")

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

def tune_hyperparameters(X_train, y_train):
    """
    Tune the hyperparameters of a Random Forest Classifier using GridSearchCV.

    Parameters:
    - X_train: Features for training
    - y_train: Labels for training

    Returns:
    - best_rf_model: The best Random Forest model found by GridSearchCV
    """
    print("[INFO] Hyperparameter tuning with GridSearchCV...")
    
    # Define the RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Apply GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    print(f"[INFO] Best Hyperparameters: {grid_search.best_params_}")
    
    # Get the best model
    best_rf_model = grid_search.best_estimator_

    # Save the best model to a file
    joblib.dump(best_rf_model, 'src/model/best_random_forest_model.joblib')
    print("[INFO] Best model saved to 'src/model/best_random_forest_model.joblib'")

    return best_rf_model
