import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from flask import Flask, jsonify, Response
from model import download_data, format_data, train_model
from config import model_file_path

from prophet import Prophet

app = Flask(__name__)


def update_data():
    """Download price data, format data and train model."""
    download_data()
    format_data()
    train_model()


def get_eth_inference():
    """Load model and predict current price."""
    print("---pred price---")
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    today_date = datetime.today().date()
    tmr_date = today_date + timedelta(days=1)
    tmr_date = tmr_date.strftime('%Y-%m-%d')

    print(f"tmr date : {tmr_date}")

    tmr_date = datetime.strptime(tmr_date,'%Y-%m-%d')

    with open('latest_date.txt', 'r') as file:
        latest_date = file.read()

    print(f"latest date from df {latest_date}")

    latest_date = datetime.strptime(latest_date,'%Y-%m-%d')

    period = (tmr_date - latest_date).days

    print(f"period = {period}")

    future = loaded_model.make_future_dataframe(periods = period)
    forcast = loaded_model.predict(future)

    print(f"pred price : {forcast['yhat'].iloc[-1]}")

    return forcast['yhat'].iloc[-1]


@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token != "ETH":
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_eth_inference()
        return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
