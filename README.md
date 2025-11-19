from setuptools import setup, find_packages

setup(
    name="predictor_5",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "tqdm",
        "yfinance",
        "arch",
        "lightgbm",
        "scikit-learn",
        "matplotlib",
        "openpyxl",
    ],
    entry_points={
        "console_scripts": [
            "fetch_data=predictor_5:fetch_data",
            "process_data=predictor_5:process_data",
            "backtest=predictor_5:backtest",
        ]
    },
    description="Stock forecasting pipeline with LSTM, TCN, LightGBM, HMM",
    author="Daksh Shah",
)


'''command for fetch_data:
    python -m src.fetch_data
'''

'''command for data_processing:
python -c "from src.data_processing import build_features; build_features('data/raw_data/deepakntr_bo_data.csv','data/processed_data/deepakntr_bo_features.csv','data/raw_data/brent.csv','data/raw_data/usd_inr.csv')"
'''


''' command run for backtesting:
 python -m src.pipeline.backtest_unified --processed data/processed_data/deepakntr_bo_features.csv --out data/backtest/backtest_results.xlsx --train_end 2025-08-30'''


commands to run everything
 after install
 pip install -e 

 and then:

fetch_data
process_data
backtest