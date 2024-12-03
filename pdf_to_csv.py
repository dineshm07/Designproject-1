import pdfplumber
import pandas as pd
import re

def extract_transactions_from_pdf(pdf_path, output_csv_path):
    # Initialize an empty list to hold transaction data
    data = {
        "Date": [],
        "Details": [],
        "Debit": [],
        "Credit": [],
        "Balance": []
    }

    with pdfplumber.open(pdf_path) as pdf:
        # Loop through pages, starting from the first page
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:  # Ensure there's text on the page
                # Split text into lines
                lines = text.split('\n')
                for line in lines:
                    # Check if the line looks like a transaction
                    if is_transaction_line(line):
                        parts = line.split()
                        if len(parts) >= 5:  # Ensure there are enough parts
                            # Extract the date using regex
                            date_match = re.search(r'(\d{1,2}\s[A-Za-z]{3}\s\d{4})', line)
                            if date_match:
                                date = date_match.group(1)  # Get the full date
                                data["Date"].append(date)
                            else:
                                data["Date"].append('')

                            data["Details"].append(" ".join(parts[3:-3]))  # Adjust index for details
                            data["Debit"].append(parts[-3])  # Debit amount
                            data["Credit"].append(parts[-2])  # Credit amount
                            data["Balance"].append(parts[-1])  # Balance amount

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Convert columns to numeric, handling errors
    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Data extracted to {output_csv_path}")

def is_transaction_line(line):
    return any(keyword in line for keyword in ["UPI", "TRANSFER", "DEBIT", "CREDIT"])  # Adjust keywords as needed
