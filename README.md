# Fraud Detection Project

## Overview

This project is a fraud detection system that leverages machine learning models to identify fraudulent transactions. It includes data preprocessing, synthetic data generation, model training, and a web-based interface for results visualization.

## Features

- Preprocessing and cleaning raw transaction data
- Generating synthetic data to enhance training datasets
- Building and training a Random Forest model
- Visualizing predictions and insights through a web dashboard

## Requirements

To run this project, you need:

- Python (>=3.9)
- Libraries specified in `req.txt`

### Install Dependencies
bash
pip install -r req.txt


## Usage

1. **Clone the Repository**
   bash
   git clone 
   cd CODE/fraud_detection_project
   

2. **Prepare the Data**
   Place your data files in the `data/` folder.

3. **Run the Web Application**
   bash
   python app.py
   

4. **Optional: Retrain the Model**
   If you need to retrain the model:
   bash
   python src/model/train_model.py
   

## Project Structure

fraud_detection_project/
├── app.py                     
├── main.py                   
├── synthetic_data.py          

data/                          
├── final_dataset.csv
├── new_transaction_data.csv
├── predictions.csv
├── transactions.csv
└── transaction_history_unlocked.pdf

src/                           
├── generate_synData.py        
├── pdf_to_csv.py              
├── predict.py                
├── visualize_predictions.py   

src/model/                     
├── best_random_forest_model.joblib 
├── train_model.py             
├── evaluate_model.py          
├── preprocessed_data.joblib   
└── preprocessor.joblib        

src/preprocess/                
└── preprocess_data.py         

static/                        
├── css/                      
└── images/                   

templates/                     
├── analysis.html
├── dashboard.html
├── index.html
└── visualize.html

req.txt                        
## How It Works

1. **Data Preprocessing**  
   The `preprocess_data.py` script cleans and prepares raw transaction data for training.

2. **Synthetic Data Generation**  
   `generate_synData.py` creates synthetic datasets to augment training data.

3. **Model Training**  
   `train_model.py` trains a Random Forest model to detect fraudulent transactions.

4. **Prediction and Visualization**  
   `predict.py` runs predictions, and the results are displayed in a user-friendly web interface.

### Libraries and Tools

This project leverages the following libraries and tools:

- *Python*: Programming language used for implementation.
- *Scikit-learn*: For machine learning model development and evaluation.
- *Pandas*: For data preprocessing and manipulation.
- *Joblib*: For saving and loading model and preprocessor artifacts.
- *Flask*: To build the web application and API endpoints.
- *Matplotlib/Seaborn*: For data visualization and exploratory analysis.

### Data Sources

- *Synthetic Dataset*: Created using generate_synData.py to supplement training data.
- *Raw Transaction Data*: Processed using scripts in the preprocess/ and data/ directories.


### Acknowledgments

- *OpenAI*: For providing tools and guidance during the project.
- *Community Contributors*: Special thanks to open-source community contributors for shared knowledge and resources.

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

---

