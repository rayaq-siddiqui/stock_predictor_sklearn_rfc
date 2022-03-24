import yfinance as yf
import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np


class Stock:
    # constructor
    def __init__(self, ticker, data_path=None):

        self.ticker = ticker
        print(self.ticker)
        if not data_path:
            self.data_path = 'data/' + str(self.ticker).lower() + '_data.json'
        else:
            self.data_path = data_path
        print(self.data_path)
        
        if os.path.exists(self.data_path):
            with open(self.data_path) as f:
                self.hist = pd.read_json(self.data_path)
        else:
            t = yf.Ticker(self.ticker)
            self.hist = t.history(period="max")
            self.hist.to_json(self.data_path)

        self.model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

        # creating the self.data object
        data = self.hist[["Close"]]
        data = data.rename(columns={"Close": "Actual_Close"})
        data["Target"] = self.hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]
        prev = self.hist.copy()
        prev = prev.shift(1)
        predictors = ["Close", "Volume", "Open", "High", "Low"]
        data = data.join(prev[predictors]).iloc[1:]
        weekly_mean = data.rolling(7).mean()["Close"]
        quarterly_mean = data.rolling(90).mean()["Close"]
        annual_mean = data.rolling(365).mean()["Close"]
        weekly_trend = data.shift(1).rolling(7).sum()["Target"]
        data["weekly_trend"] = weekly_trend
        data["weekly_mean"] = weekly_mean / data["Close"]
        data["quarterly_mean"] = quarterly_mean / data["Close"]
        data["annual_mean"] = annual_mean / data["Close"]
        data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
        data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
        data["open_close_ratio"] = data["Open"] / data["Close"]
        data["high_close_ratio"] = data["High"] / data["Close"]
        data["low_close_ratio"] = data["Low"] / data["Close"]
        predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]

        self.data = data
        self.predictors = predictors


    def backtest(self, data, model, predictors, start=1000, step=750):
        predictions = []
        # Loop over the dataset in increments
        for i in range(start, data.shape[0], step):
            # Split into train and test sets
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            
            # Fit the random forest model
            model.fit(train[predictors], train["Target"])
            
            # Make predictions
            preds = model.predict_proba(test[predictors])[:,1]
            preds = pd.Series(preds, index=test.index)
            preds[preds > .6] = 1
            preds[preds <= .6] = 0
            
            # Combine predictions and test values
            combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
            
            predictions.append(combined)
        
        return pd.concat(predictions)


    # create a prediction
    def predict(self):
        preds = self.backtest(
            self.data.iloc[365:], 
            self.model, 
            self.predictors
        )
        return preds


    # precision
    def precision(self, preds):
        return precision_score(preds["Target"], preds["Predictions"])


    # plot the last x days with preds
    def plot_preds(self, preds, x):
        return preds.iloc[-x:].plot()
        

    def plot_hist(self):
        return self.hist.plot.line(y="Close", use_index=True)