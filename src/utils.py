import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import numpy as np
import pandas as pd

# Augmented Dickey-Fuller
def do_adf_regression(df, ticker1, ticker2):
    x = df[ticker1].values
    y = df[ticker2].values

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    alpha, beta = model.params
    residuals = y - (alpha + beta * x[:, 1])

    if np.isnan(residuals).any():
        return np.nan

    t_stat, p_value, *rest = adfuller(model.resid)
    critical_values = rest[2]
    probability = np.nan
    if p_value < 0.01 and t_stat < critical_values['1%']:
        probability = 99
    elif p_value < 0.05 and t_stat < critical_values['5%']:
        probability = 95
    elif p_value < 0.1 and t_stat < critical_values['10%']:
        probability = 90

    return probability


# Augmented Engle-Granger
def do_aeg_regression(df, ticker1, ticker2):
    x = df[ticker1].values
    y = df[ticker2].values

    t_stat, p_value, critical_values = coint(x, y)

    probability = np.nan
    if p_value < 0.01 and t_stat < critical_values[0]:
        probability = 99
    elif p_value < 0.05 and t_stat < critical_values[1]:
        probability = 95
    elif p_value < 0.1 and t_stat < critical_values[2]:
        probability = 90

    return probability


def calculate_sharpe_ratio(training_set: pd.DataFrame, ticker1: str, ticker2: str, open_threshold: float, close_threshold: float):
    hedge_ratio = sm.OLS(training_set[ticker1], training_set[ticker2]).fit().params[0]

    spread = training_set[ticker1] - hedge_ratio * training_set[ticker2]
    spread_mean = spread.mean()
    spread_std = spread.std()

    calc_df = pd.DataFrame(index=training_set.index)
    calc_df['zscore'] = (spread - spread_mean) / spread_std

    calc_df['pos_1_long'] = 0
    calc_df['pos_2_long'] = 0
    calc_df['pos_1_short'] = 0
    calc_df['pos_2_short'] = 0
    calc_df.loc[calc_df.zscore >= open_threshold, ('pos_1_short', 'pos_2_short')] = [ -1, 1 ] # Short spread
    calc_df.loc[calc_df.zscore <= -open_threshold, ('pos_1_long', 'pos_2_long')] = [ 1, -1 ] # Long spread
    calc_df.loc[calc_df.zscore <= close_threshold, ('pos_1_short', 'pos_2_short')] = 0 # Close position short
    calc_df.loc[calc_df.zscore >= -close_threshold, ('pos_1_long', 'pos_2_long')] = 0 # Close position long

    calc_df.ffill(inplace=True)

    longs = calc_df.loc[:, ('pos_1_long', 'pos_2_long')]
    shorts = calc_df.loc[:, ('pos_1_short', 'pos_2_short')]

    positions = np.array(longs) + np.array(shorts)
    positions = pd.DataFrame(positions)

    longs_with_dates = calc_df.loc[:, ('pos_1_long', 'pos_2_long')].copy()
    longs_with_dates.columns = [ticker1, ticker2]

    shorts_with_dates = calc_df.loc[:, ('pos_1_short', 'pos_2_short')].copy()
    shorts_with_dates.columns = [ticker1, ticker2]

    positions_with_dates = longs_with_dates + shorts_with_dates
    dailyret = training_set.loc[:, (ticker1, ticker2)].pct_change()

    pnl = (np.array(positions.shift()) * np.array(dailyret)).sum(axis=1)
    sharpe_ration = np.sqrt(252) * pnl[1:].mean() / pnl[1:].std()

    return (sharpe_ration, positions_with_dates)


def calculate_sharpe_ratio(df, ticker1, ticker2, hedge_ration, open_threshold, close_threshold):
    spread = df[ticker1] - hedge_ration * df[ticker2]
    spread_mean = spread.mean()
    spread_std = spread.std()
    
    df['zscore'] = (spread - spread_mean) / spread_std
    df['pos_1_long'] = 0
    df['pos_2_long'] = 0
    df['pos_1_short'] = 0
    df['pos_2_short'] = 0

    df.loc[df.zscore >= open_threshold, ('pos_1_short', 'pos_2_short')] = [ -1, 1 ] # Short spread
    df.loc[df.zscore <= -open_threshold, ('pos_1_long', 'pos_2_long')] = [ 1, -1 ] # Long spread
    df.loc[df.zscore <= close_threshold, ('pos_1_short', 'pos_2_short')] = 0 # Close position short
    df.loc[df.zscore >= -close_threshold, ('pos_1_long', 'pos_2_long')] = 0 # Close position long

    df.ffill(inplace=True) # ensure existing positions are carried forward unless there is an exit signal

    longs = df.loc[:, ('pos_1_long', 'pos_2_long')]
    shorts = df.loc[:, ('pos_1_short', 'pos_2_short')]
    positions = np.array(longs) + np.array(shorts)
    positions = pd.DataFrame(positions)

    longs_with_dates = df.loc[:, ('pos_1_long', 'pos_2_long')].copy()
    longs_with_dates.columns = [ticker1, ticker2]

    shorts_with_dates = df.loc[:, ('pos_1_short', 'pos_2_short')].copy()
    shorts_with_dates.columns = [ticker1, ticker2]

    positions_with_dates = longs_with_dates + shorts_with_dates

    new_columns = pd.MultiIndex.from_product([[f"{ticker1}_{ticker2}"], positions_with_dates.columns])
    positions_with_dates.columns = new_columns

    dailyret = df.loc[:, (ticker1, ticker2)].pct_change()
    pnl = (np.array(positions.shift()) * np.array(dailyret)).sum(axis=1)

    sharpe_ratio = np.sqrt(252) * pnl[1:].mean() / pnl[1:].std()

    return (sharpe_ratio, spread_mean, spread_std, positions_with_dates)