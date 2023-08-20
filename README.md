# data_science_practicum_2
Machine Learning Models - Cryptocurrency Price Prediction Analysis
ARIMA, LSTM, SMA and EMA models fitted against Ethereum cryptocurrency price training and test datasets.

This repository contains Python code for analyzing Ethereum (ETH) price data using ARIMA and LSTM models. 
The code downloads historical price data, performs exploratory data analysis (EDA), and then applies both ARIMA and LSTM models to make price predictions.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Requirements](#requirements)
- [Usage](#usage)
  - [ARIMA Model](#arima-model)
  - [LSTM Model](#lstm-model)
  - [Simple Moving Average](#simple-moving-average)
  - [Exponential Moving Average](#exponential-moving-average)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Ethereum (ETH) is a popular cryptocurrency, and analyzing its price data can provide valuable insights for investors and traders. This project uses Python to analyze historical ETH price data and make predictions using two different approaches: ARIMA and LSTM models.

## Getting Started
### Prerequisites
Before you begin, ensure you have the following prerequisites installed:

- Python 3.6+
- Jupyter Notebook (for running the code)
- Required Python libraries (e.g., yfinance, matplotlib, numpy, statsmodels, tensorflow)

### Installation
- Clone this repository:
```bash
git clone https://github.com/ksvec/data_science_practicum2.git
```
- Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Requirements
#### yfinance
  - This library is used to download historical price data for Ethereum (ETH) from Yahoo Finance.
#### datetime
  - It's a built-in Python module used to work with dates and times. In your script, it's used to handle date-related operations.
#### pandas
  - Pandas is a powerful data manipulation and analysis library. In your script, it's used to store and manipulate data, especially for creating dataframes.
#### matplotlib.pyplot
  - This library is used for creating data visualizations such as line charts and plots.
#### numpy
  - NumPy is used for numerical operations and array manipulations.
#### statsmodels
  - Statsmodels is a library for estimating and interpreting statistical models. In your script, it's used for implementing the ARIMA model.
#### math
  - The built-in math module provides mathematical functions. In your script, it's used for mathematical operations.
#### sklearn.metrics
  - This library provides various metrics for evaluating the performance of machine learning models. In your script, it's used to calculate RMSE (Root Mean Squared Error) and MAPE (Mean Absolute Percentage Error).
#### sklearn.model_selection
  - This library provides tools for model selection and evaluation, including train-test splits. In your script, it's used for splitting the data into training and testing sets.
#### tensorflow
  - TensorFlow is an open-source machine learning framework. In your script, it's used for building and training LSTM (Long Short-Term Memory) models for time-series forecasting.
#### pandas_profiling (commented out)
  - This library generates an EDA (Exploratory Data Analysis) report for your dataset. In your script, it's not used but seems to be included in comments.

## Usage
### ARIMA Model
The ARIMA model is applied to the ETH price data to make time-series predictions. You can run the ARIMA analysis by following the Jupyter Notebook provided in the repository.

### LSTM Model
The LSTM model is used to make long-term predictions for ETH prices. The LSTM code can also be found in the Jupyter Notebook.

### Simple Moving Average
This project includes a Simple Moving Average (SMA) analysis. It calculates and visualizes the SMA of ETH prices.

### Exponential Moving Average
The Exponential Moving Average (EMA) analysis is also included. It calculates and visualizes the EMA of ETH prices.

### Results
The results of the analysis, including visualizations and performance metrics (RMSE and MAPE), can be found in the Jupyter Notebook.

## Contributing
If you would like to contribute to this project or report issues, please follow these guidelines:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and test them.
Create a pull request, describing the changes you made.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
