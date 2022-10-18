import pandas_datareader as pdr
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tickers = ['AAPL', 'MSFT', 'TWTR', 'IBM']
start = dt.datetime(2021, 1, 1)
data = pdr.get_data_yahoo(tickers, start)
data = data['Adj Close']

# log returns
log_returns = np.log(data/data.shift())

# Monte Carlo Simulation, code will run 5000 experiments
n = 5000
weights = np.zeros((n, 4))
exp_rtns = np.zeros(n)
exp_vols = np.zeros(n)
sharpe_ratios = np.zeros(n)
for i in range(n):
    weight = np.random.random(4)
    weight /= weight.sum()
    weights[i] = weight

    exp_rtns[i] = np.sum(log_returns.mean() * weight) * 252
    exp_vols[i] = np.sqrt(np.dot(weight.T, np.dot(log_returns.cov() * 252, weight)))
    sharpe_ratios[i] = exp_rtns[i] / exp_vols[i]


argmax = sharpe_ratios.argmax()
print(data)
print(weights[argmax])

# visualisation of the data
fig, ax = plt.subplots()
ax.scatter(exp_vols, exp_rtns, c=sharpe_ratios)
ax.scatter(exp_vols[sharpe_ratios.argmax()], exp_rtns[sharpe_ratios.argmax()], c='r')
ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
plt.show()

