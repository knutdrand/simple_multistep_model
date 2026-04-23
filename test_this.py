from pathlib import Path

import pandas as pd

from train import train
from predict import predict

def test_pipeline():
    filename = Path(__file__).parent / "training_data.csv"
    future_filename = Path(__file__).parent / "future_data.csv"
    historic_filename = Path(__package__) / "historic_data.csv"
    train(filename, "tmp.pkl")
    predict('tmp.pkl', historic_filename, future_filename, 'tmp_pred.csv')