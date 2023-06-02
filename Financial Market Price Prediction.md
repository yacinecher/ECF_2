Project Title

Financial Market Price Prediction

Description

This project aims to predict financial market prices using machine learning techniques. The dataset used contains historical market data, including open, high, low, close prices, volume, and adjusted close prices. The goal is to build predictive models based on various features and evaluate their performance.

Features

Preprocessing: The dataset is preprocessed by converting the 'Date' column to datetime format and sorting the data chronologically.
Feature Engineering: Additional features are created, including rolling means and standard deviations of opening prices, ratios between short-term and long-term means/std, and date-based features (year, month, day).
Model Training and Evaluation: Two machine learning models, Linear Regression and K-Nearest Neighbors, are trained and evaluated using mean squared error as the performance metric.
Results: The root mean squared error (RMSE) is calculated for both models to assess their prediction accuracy.
Dependencies

Python 
Pandas 
NumPy 
Scikit-learn 
Instructions

Clone the repository: git clone <repository_url>
Install the required dependencies: pip install -r requirements.txt
Run the main script: python main.py
View the results and performance metrics in the console output.
Contributing

Contributions to this project are welcome. Feel free to open issues or submit pull requests for any improvements or bug fixe