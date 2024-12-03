import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # To save preprocessor

# Step 1: Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Step 2: Data Preprocessing
def preprocess_data(df, is_predicting=False):
    """
    Preprocess the data for model training or prediction.

    Parameters:
    - df: DataFrame containing the data to preprocess
    - is_predicting: Whether the function is preprocessing for prediction (defaults to False for training)

    Returns:
    - X_processed: Processed feature data
    - y (optional): Target variable for training
    """
    if not is_predicting:
        # For training data, separate features (X) and target (y)
        X = df.drop(columns=["Fraudulent", "Date", "Transaction_Time"])  # Dropping non-features
        y = df["Fraudulent"]
        return preprocess_features(X), y
    else:
        # For prediction data, only preprocess features (no target column)
        X = df.drop(columns=["Date", "Transaction_Time"])  # Dropping non-features for prediction
        return preprocess_features(X)


def preprocess_features(X):
    """
    Apply preprocessing steps to the features.

    Parameters:
    - X: Features to preprocess

    Returns:
    - X_processed: Processed feature data
    """
    # Define numerical and categorical columns
    numerical_cols = ["Debit", "Credit", "Balance", "Transaction_Frequency", "Hour_of_Day"]
    categorical_cols = ["Details", "Location", "Transaction_Type"]

    # Define preprocessing for numerical features (StandardScaler)
    numerical_transformer = StandardScaler()

    # Define preprocessing for categorical features (OneHotEncoder)
    categorical_transformer = OneHotEncoder(drop='first')  # Drop first to avoid dummy variable trap

    # Create a column transformer that applies the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create a pipeline that applies preprocessing to the data
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Apply the pipeline to the features (X)
    X_processed = pipeline.fit_transform(X)

    # Save the preprocessor for future use (e.g., when making predictions)
    joblib.dump(pipeline, 'src/model/preprocessor.joblib')

    return X_processed

# Step 3: Split data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Main execution for testing
if __name__ == "__main__":
    data_path = r"E:\DESIGN PROJECT 1\fraud_detection_project\data\final_dataset.csv"
    df = load_data(data_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data preprocessing complete and split into train and test sets.")
