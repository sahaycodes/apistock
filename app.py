from flask import Flask, request, jsonify, render_template
import yfinance as yf
from pandas_datareader import data as web
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import plotly.express as px

app = Flask(__name__)

def get_data(assets):
    df = pd.DataFrame()
    yf.pdr_override()
    for stock in assets:
        df[stock] = web.get_data_yahoo(stock)['Adj Close']
    return df

def optimize_portfolio(df, tickers_string, starting_amount=100):
    corr_df = df.corr().round(2)
    fig_corr = px.imshow(corr_df, text_auto=True)
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=0.02)

    weights = ef.clean_weights()
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
    weights_df = pd.DataFrame.from_dict(weights, orient='index')
    weights_df.columns = ['Weights']
    weights_df.index.name = "Ticker"

    df['Optimized Portfolio'] = 0
    for ticker, weight in weights.items():
        df['Optimized Portfolio'] += df[ticker] * weight

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=starting_amount)

    allocation, leftover = da.greedy_portfolio()

    returns_optimized = plot_chart(df['Optimized Portfolio'], title=None)

    port_Returns = (expected_annual_return * 100).round(2)

    return weights_df

def plot_chart(df, title):
    fig = px.line(df, title=title)
    return fig

def gaussian_pdf(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def calculate_probabilities(user_point, centroids):
    distances = np.array([euclidean_distance(user_point, centroid) for centroid in centroids])
    std = np.std(distances)
    probabilities = np.array([gaussian_pdf(dist, 0, std) for dist in distances])
    normalized_probabilities = probabilities / np.sum(probabilities)
    return normalized_probabilities

def remove_spaces(text):
    return re.sub(r'\s+', '', text)

@app.route('/', methods=['GET', 'POST'])
def optimize():
    if request.method == 'POST':
        data = request.form

        lifestyle_risk = int(data['lifestyle_risk'])
        expected_annual_roi = float(data['expected_annual_roi'])
        principal_amount = float(data['principal_amount'])

        info_data = [
            {"Symbol": "AAPL", "Annualized ROI": 17.624598, "Volatility": 0.027930},
            {"Symbol": "AMZN", "Annualized ROI": 30.874074, "Volatility": 0.035528},
            {"Symbol": "ARKK", "Annualized ROI": 7.464424, "Volatility": 0.023803},
            {"Symbol": "BABA", "Annualized ROI": -1.149219, "Volatility": 0.026288},
            {"Symbol": "BTC-USD", "Annualized ROI": 57.557349, "Volatility": 0.036763},
            {"Symbol": "ELF", "Annualized ROI": 21.738374, "Volatility": 0.031783},
            {"Symbol": "ETH-USD", "Annualized ROI": 35.917395, "Volatility": 0.046763},
            {"Symbol": "GC=F", "Annualized ROI": 8.950140, "Volatility": 0.010886},
            {"Symbol": "GOOGL", "Annualized ROI": 22.442833, "Volatility": 0.019344},
            {"Symbol": "GSK", "Annualized ROI": 10.011961, "Volatility": 0.017046},
            {"Symbol": "ITC.NS", "Annualized ROI": 16.263788, "Volatility": 0.020341},
            {"Symbol": "JNJ", "Annualized ROI": 10.925850, "Volatility": 0.014432},
            {"Symbol": "MSFT", "Annualized ROI": 24.020376, "Volatility": 0.021158},
            {"Symbol": "NFLX", "Annualized ROI": 31.412082, "Volatility": 0.035304},
            {"Symbol": "NVDA", "Annualized ROI": 34.711791, "Volatility": 0.037871},
            {"Symbol": "PLD", "Annualized ROI": 5.721513, "Volatility": 0.023028},
            {"Symbol": "QCOM", "Annualized ROI": 18.908380, "Volatility": 0.030630},
            {"Symbol": "SQ", "Annualized ROI": 17.814773, "Volatility": 0.036852},
            {"Symbol": "TCEHY", "Annualized ROI": 17.299087, "Volatility": 0.023695},
            {"Symbol": "TSLA", "Annualized ROI": 37.054781, "Volatility": 0.035839},
            {"Symbol": "XOM", "Annualized ROI": 7.051585, "Volatility": 0.014561}
        ]

        info = pd.DataFrame(info_data)

        model_data = info[['Annualized ROI', 'Volatility']]
        kmeans = KMeans(n_clusters=3, random_state=4)
        kmeans.fit(model_data)
        clusters = kmeans.predict(model_data)
        info['Cluster'] = clusters
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(model_data, clusters)
        centroids = kmeans.cluster_centers_

        if lifestyle_risk == 0:
            expected_volatility = centroids[0][1]
        elif lifestyle_risk == 1:
            expected_volatility = centroids[2][1]
        elif lifestyle_risk == 2:
            expected_volatility = centroids[1][1]
        else:
            return jsonify({"error": "Invalid lifestyle risk value"})

        model_input = np.array([[expected_annual_roi, expected_volatility]])
        predicted_cluster = knn.predict(model_input)

        probabilities = calculate_probabilities(model_input, centroids)
        nearest_centroid_index = np.argmax(probabilities)
        weighted_amounts = principal_amount * probabilities
        weights_df = pd.DataFrame({'Weight': weighted_amounts})

        clusters_data = {
            "Symbols": [
                "ARKK, BABA, GC=F, GSK, JNJ, PLD, XOM",
                "AMZN, BTC-USD, ETH-USD, NFLX, NVDA, TSLA",
                "AAPL, ELF, GOOGL, ITC.NS, MSFT, QCOM, SQ, TCEHY"
            ]
        }

        clusters_df = pd.DataFrame(clusters_data)
        clusters_df['Weights'] = weights_df['Weight']

        results = []
        for index, row in clusters_df.iterrows():
            ticker = str(row['Symbols'])
            ticker = remove_spaces(ticker)
            assets = ticker.split(',')
            df = get_data(assets)
            starting_amount = row['Weights']
            weights_df = optimize_portfolio(df, ticker, starting_amount=starting_amount)
            results.append({
                "Symbols": row['Symbols'],
                "Weights": weights_df.to_dict()
            })

        return jsonify({"results": results, "clusters": clusters_df.to_dict(orient='records')})
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
