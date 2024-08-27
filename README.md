# CarDekho - Used Car Price Prediction

## Overview
This project focuses on predicting the prices of used cars listed on CarDekho, one of India’s leading car marketplaces. The project utilizes machine learning techniques to build an accurate pricing model based on various features of the cars. The goal is to enhance the customer experience by providing accurate price predictions and assisting CarDekho in streamlining its pricing process.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Model Development](#model-development)
7. [Model Evaluation](#model-evaluation)
8. [Model Optimization](#model-optimization)
9. [Deployment](#deployment)
10. [Usage](#usage)
11. [Results](#results)
12. [Challenges](#challenges)
13. [Future Work](#future-work)


## Project Structure

The project directory contains the following files and folders:

- **`xlsx Files/`**: Directory containing datasets
  - `delhi.xlsx` - Data from Delhi
  - `bangalore.xlsx` - Data from Bangalore
  - `chennai.xlsx` - Data from Chennai
  - `kolkata.xlsx` - Data from Kolkata
  - `hyderabad.xlsx` - Data from Hyderabad
  - `jaipur.xlsx` - Data from Jaipur

- **`scripts/`**: Python scripts for preprocessing, modeling, etc.
  - `preprocess.py` - Data preprocessing script
  - `train_model.py` - Model training script

- **`streamlit_app.py`**: Streamlit application file for interacting with the model

- **`best_model_pipeline.pkl`**: The best performing model, saved for deployment

- **`README.md`**: Project documentation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/arunachaladevaraj/used-car-price-prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd used-car-price-prediction
    ```

3. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset
The dataset comprises six Excel files representing car data from different cities (Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Jaipur). Features include:
- Car Model
- Year of Manufacture
- Kilometers Driven
- Transmission Type (Manual/Automatic)
- Fuel Type (Petrol/Diesel/CNG)
- Mileage
- Insurance Validity
- Ownership History
- City

## Data Preprocessing
### Steps Involved:
- **Data Cleaning:**
  - Removal of duplicates and irrelevant data.
  - Handling missing values using various imputation techniques.
  - Normalization and standardization of numerical data.
- **Feature Engineering:**
  - Derived features like Car Age.
  - Conversion of categorical variables into numerical values using one-hot encoding.
- **Dealing with Outliers:**
  - Identification and removal of outliers using statistical methods.
- **Encoding Categorical Data:**
  - Categorical encoding for Ownership History and Insurance Validity.

### Tools Used:
- Pandas for data manipulation.
- NumPy for numerical computations.
- Scikit-learn for preprocessing tasks.

## Exploratory Data Analysis
EDA was conducted to identify patterns and insights within the data:
- **Price Distribution Analysis:** Analyzing car price distributions across cities.
- **Correlation Analysis:** Examining correlations between features and car price.
- **Feature Importance:** Identifying key features affecting car prices.
- **Visualizations:** Using Matplotlib and Seaborn to create data visualizations.

### Key Insights:
- Cars with automatic transmission tend to be priced higher.
- Diesel cars generally have lower resale values compared to petrol cars.
- The price drops significantly after the car’s fifth year of use.

## Model Development
### Models Considered:
- **Linear Regression:** Baseline model for performance.
- **Random Forest Regressor:** Handles non-linear relationships well.
- **XGBoost:** Known for high performance and efficiency.
- **Gradient Boosting Regressor:** Improves performance iteratively.

### Model Selection:
- Random Forest Regressor and XGBoost were found to be the best-performing models.

### Tools Used:
- Scikit-learn for Linear Regression and Random Forest.
- XGBoost library for XGBoost model.

## Model Evaluation
### Evaluation Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

### Performance:
- XGBoost emerged as the best-performing model with the lowest RMSE and highest R² score.

## Model Optimization
Hyperparameter tuning was performed using GridSearchCV:
- Number of Trees (n_estimators)
- Maximum Depth (max_depth)
- Learning Rate

The optimized model was saved as `best_model_pipeline.pkl` for deployment.

## Deployment
The final model was deployed using Streamlit, allowing users to interact with the model via a web interface.

### Streamlit App Features:
- Transmission: Select between 'Automatic' and 'Manual'.
- Year of Manufacture: Choose from 2000 to 2024.
- Model Year: Select a year from 1980 to 2024.
- City: Choose from six cities.
- Insurance Validity: Options include 'Third Party insurance', 'Zero Dep', 'Comprehensive'.
- Ownership: Choose between various ownership options.
- Fuel Type: Options include 'CNG', 'Petrol', 'Diesel'.
- Kilometers Driven: Slider between 10,000 and 200,000 km.
- Body Type: Select from options like 'Hatchback', 'Sedan', etc.
- Mileage: Slider between 10 and 30 km/l.

### Running the App:
1. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```
2. Fill in the required fields in the sidebar to get predictions.

## Usage
This project helps CarDekho and its users in making informed decisions about used car prices through an interactive Streamlit application.

## Results
The final model demonstrates strong predictive power with an R² score of Random Forest. The Streamlit app provides quick and accurate predictions.

## Challenges
- **Data Cleaning:** Handling missing and inconsistent data.
- **Feature Engineering:** Experimenting with features.
- **Model Optimization:** Time-consuming hyperparameter tuning.

## Future Work
- **Incorporate Additional Features:** Adding more features like service history.
- **Expand Dataset:** Include data from more cities.
- **Enhance Streamlit App:** Improve UI/UX and add more interactive features.
