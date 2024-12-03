import pandas as pd
import numpy as np


def generate_synthetic_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    print("Initial Data:")
    print(df.head())  # Debugging: Check initial data

    df.columns = df.columns.str.strip()
    df['Date'] = df['Date'].astype(str).str.strip()

    # Convert 'Date' using the correct format
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y', errors='coerce')
    except Exception as e:
        print(f"Error parsing dates: {e}")

    print("NaT values in Date column:")
    print(df[df['Date'].isnull()])  # Print rows with NaT values

    df = df.dropna(subset=['Date'])
    print("Data after dropping NaT values:")
    print(df.head()) 
    
    df = df.sort_values(by=['Date'])
    df['Transaction_Frequency'] = df.groupby(df['Date'].dt.date)['Date'].transform('count') - 1
    df['Transaction_Time'] = df['Date'] + pd.to_timedelta(np.random.randint(0, 24), unit='h')
    df['Hour_of_Day'] = df['Transaction_Time'].dt.hour

    locations = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Trichy", "Kolkata", "International"]
    df['Location'] = np.where(np.random.rand(len(df)) > 0.8, "International", np.random.choice(locations, len(df)))
    df['Transaction_Type'] = df['Details'].apply(lambda x: "UPI" if "UPI" in x else "ATM" if "ATM" in x else "Other")

    df['Fraudulent'] = np.where((df['Debit'] > 1000) & (df['Location'] == "International") & (df['Hour_of_Day'] > 22), 1, 0)

    if df.empty:
        print("Warning: The final DataFrame is empty. No data will be saved.")
    else:
        print("Final DataFrame before saving:")
        print(df.head())  # Ensure there is data before saving
        df.to_csv(output_csv, index=False)
        print(f"Synthetic features added to {output_csv}")
