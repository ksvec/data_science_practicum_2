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
- [Usage](#usage)
  - [ARIMA Model](#arima-model)
  - [LSTM Model](#lstm-model)
  - [Simple Moving Average](#sma-model)
  - [Exponential Moving Average](#ema-model)
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
