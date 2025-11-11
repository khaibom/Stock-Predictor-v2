# Stock-Predictor

This project builds an LSTM-based machine learning model to predict whether NVIDIA’s stock closing price will go up or down next week using historical stock data and technical indicators like MACD, RSI, and SMA.

## Python Version Requirement

This project requires **Python 3.10.x**

TensorFlow does not support Python 3.13 as of this writing.

### Check your version:

```bash
python --version
```

### If you're using Python 3.13 or newer, install Python 3.10 using pyenv:
```bash
pyenv install 3.10.13
pyenv local 3.10.13
~/.pyenv/versions/3.10.13/bin/python -m venv .venv
```

## Setup Instructions

### Linux/macOS
```bash
bash setup.sh
source .venv/bin/activate
```

### Windows
```bash
setup.bat
.venv\Scripts\activate
```

## Project Structure
```
Stock-Predictor/
├── data/
│   ├── raw/                   # Raw stock data
│   └── processed/             # Data with engineered features
├── ml_model/
│   ├── collect_data.py        # Script to download and save raw stock data
│   ├── train_model.ipynb      # Notebook for model development
│   ├── train.py               # Training script
│   ├── model.h5               # Saved LSTM model
│   ├── scaler.pkl             # Feature scaler
│   └── preprocess.py          # Data prep functions(e.g.,sliding window)
|   ├── feature_engineering/
│       ├── lag_features.py
│       ├── technical_indicators.py
│       └── volatility_features.py
├── predictor/
│   └── predictor.py           # Load model and make predictions
├── streamlit_app/
│   └── app.py                 # Streamlit frontend
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works
1. Pull daily historical stock data using yfinance
2. Calculate MACD, RSI, SMA, and EMA features
3. Generate target column: 1 if next week's close > this week's, else 0
4. Normalize and reshape data into sliding window format for LSTM
5. Train and evaluate an LSTM-based binary classifier
6. Deploy predictions via Streamlit interface

## Usage

### Train Model
### Run app

## Dependencies
All required libraries are listed in requirements.txt.

Main dependencies:
- pandas, numpy
- yfinance (data collection)
- ta (technical indicators)
- scikit-learn, tensorflow (ML)
- streamlit, plotly, matplotlib, seaborn (visualization)

## Contributors
[Phuong Anh Nguyen](https://github.com/npa02)
[Dinh Khai Tran](https://github.com/khaibom)
