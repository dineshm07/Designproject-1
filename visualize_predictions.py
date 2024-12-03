import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_predictions(file_path):
    print(f"[INFO] Loading predictions from {file_path}...")
    predictions_df = pd.read_csv(file_path)
    print("[INFO] Columns in predictions file:", predictions_df.columns)  # Print columns
    return predictions_df

def save_fraudulent_distribution(predictions_df, output_path):
    print("[INFO] Saving fraudulent distribution plot...")
    fraud_count = predictions_df['Fraudulent_Prediction'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=fraud_count.index, y=fraud_count.values, palette='viridis')
    plt.title('Predicted Fraudulent vs Non-Fraudulent Transactions')
    plt.xlabel('Fraudulent (1) / Non-Fraudulent (0)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'], rotation=0)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the plot as a PNG file
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Plot saved to {output_path}.")

def main():
    predictions_file = 'data/predictions.csv'  # Update path if necessary
    predictions_df = load_predictions(predictions_file)

    # Update this line to use 'Fraudulent_Prediction' instead of 'Predicted_Fraudulent'
    if 'Fraudulent_Prediction' in predictions_df.columns:
        output_path = 'static/images/fraudulent_distribution.png'
        save_fraudulent_distribution(predictions_df, output_path)
    else:
        print("[ERROR] 'Fraudulent_Prediction' column not found in the predictions file.")

if __name__ == "__main__":
    main()
