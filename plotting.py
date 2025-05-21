# === plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from functions import get_log_return_df, get_price_df, FIVE_YRS, NOW

def plot_df(df,title="Quick Plot", ypct = True, download_graph=False):
    """Quickly plots a df"""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    plt.figure(figsize=(10, 6))
    plt.plot(df, color='#0e5690')
    plt.title(title, fontsize=14, fontfamily='serif')
    plt.xlabel('Date', fontsize=12, fontfamily='serif', fontstyle='italic')
    plt.ylabel('Values', fontsize=12, fontfamily='serif', fontstyle='italic')
    plt.xticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.yticks(fontfamily='serif', fontstyle='normal', rotation=45)
    if ypct:
        plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))

    if download_graph:
        plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()
def show_rolling_volatility_graph(ticker, days:int = 360, start=FIVE_YRS, end=NOW, download_graph=False):
    """Shows the rolling volatility graph"""
    return_df = pd.DataFrame(get_log_return_df(ticker))

    rolling_volatility = return_df.rolling(window=days).std() * np.sqrt(252)
    rolling_volatility = rolling_volatility.loc[start:end]

    plot_df(rolling_volatility,f"{ticker} {days}-Day Rolling Volatility",download_graph=download_graph)
def show_rolling_volatility_comparison(tickers, highlights=[], days:int = 360, start=FIVE_YRS, end=NOW):
    """Shows the rolling volatility graph for multiple tickers"""
    return_df = pd.DataFrame(get_log_return_df(tickers))

    rolling_volatility = return_df.rolling(window=days).std() * np.sqrt(252)
    rolling_volatility = rolling_volatility.loc[start:end]

    if isinstance(rolling_volatility, pd.Series):
        rolling_volatility = rolling_volatility.to_frame()

    # Highlight lines in red
    colors = {col: '#0e5690' if col not in highlights else 'red' for col in rolling_volatility.columns}

    # Plot non-highlighted lines first
    for col in rolling_volatility.columns:
        if col not in highlights:
            plt.plot(rolling_volatility.index, rolling_volatility[col], color=colors[col], label=col, zorder=1)

    # Plot highlighted lines on top
    for col in highlights:
        plt.plot(rolling_volatility.index, rolling_volatility[col], color='red', label=col, linewidth=2, zorder=2)

    # Formatting
    plt.title(f"{days}-Day Rolling Volatility Comparison", fontsize=14, fontfamily='serif')
    plt.xticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.yticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.xlabel('Date', fontsize=12, fontfamily='serif', fontstyle='italic')
    plt.ylabel('Values', fontsize=12, fontfamily='serif', fontstyle='italic')

    plt.show()
def plot_beta_scatter(ticker, benchmark='^GSPC', freq='ME', start=FIVE_YRS, end=NOW, title="Portfolio", download_graph=False):
    """Plots the portfolio to benchmark return scatter plot with lines of best fit"""
    return_df = get_log_return_df(ticker,freq,start,end).squeeze()
    market_df = get_log_return_df(benchmark,freq,start,end).squeeze()

    x = market_df
    y = return_df
    m, b = np.polyfit(x, y, 1)

    plt.scatter(x, y, alpha=0.7, label='Returns', color='#156082')

    plt.plot(x, m*x + b, color='#0E3C51', label=f'Best Fit Line (y = {m:.2f}x + {b:.2f})')

    x_mid = np.mean(x)
    y_mid = m * x_mid + b
    neg_x = x[x < 0]
    neg_y = y[x < 0]
    pos_x = x[x >= 0]
    pos_y = y[x >= 0]

    m_neg, b_neg = np.polyfit(neg_x, neg_y, 1)
    m_pos, b_pos = np.polyfit(pos_x, pos_y, 1)

    plt.plot(neg_x, m_neg * neg_x + b_neg, color='#B84848', label=f'Negative X Best Fit (y = {m_neg:.2f}x + {b_neg:.2f})')
    plt.plot(pos_x, m_pos * pos_x + b_pos, color='#5BB848', label=f'Positive X Best Fit (y = {m_pos:.2f}x + {b_pos:.2f})')

    plt.xlabel('Market')
    plt.ylabel(title)

    padding = 0.01
    plt.grid(True)
    x_range = max(x.max() - x.min(), y.max() - y.min()) + 2 * padding
    x_mid = (x.max() + x.min()) / 2
    y_mid = (y.max() + y.min()) / 2
    plt.xlim(x_mid - x_range / 2, x_mid + x_range / 2)
    plt.ylim(y_mid - x_range / 2, y_mid + x_range / 2)

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.yticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.xlabel('Market', fontsize=12, fontfamily='serif', fontstyle='italic')
    plt.ylabel(title, fontsize=12, fontfamily='serif', fontstyle='italic')
    plt.text(
        0.02, 0.98,
        f"β: {m:.2f}\nβ⁺: {m_pos:.2f}\nβ⁻: {m_neg:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10, fontfamily='serif', verticalalignment='top',
        bbox=dict(facecolor='#FFFBC6')
    )

    if download_graph:
        plt.savefig(f'{title}_beta_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
def plot_stock_returns(tickers, start=FIVE_YRS, end=NOW):
    df = get_price_df(tickers, start=start, end=end)
    df = df / df.iloc[0] - 1

    # Generate a color palette with seaborn
    colors = sns.color_palette("Blues_d", n_colors=len(df.columns))

    plt.figure(figsize=(10, 5))

    for i, column in enumerate(df.columns):
        # Plot each stock with a unique color from the palette
        plt.plot(df.index, df[column], label=column, color=colors[i])

        # Calculate the total return (final value - initial value) for the label
        total_return = (df[column].iloc[-1] - df[column].iloc[0]) * 100

        # Add a small label with the total return at the end of each line
        plt.text(df.index[-1], df[column].iloc[-1], f'{total_return:.2f}%',
                color=colors[i], ha='left', va='center', fontsize=10, fontfamily='serif')

    # Add reference line at 0%
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    plt.legend(prop={'family': 'serif'})
    plt.title('Stock Performance', fontsize=14, fontfamily='serif')
    plt.xticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.yticks(fontfamily='serif', fontstyle='normal', rotation=45)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.xlabel('Date', fontsize=12, fontfamily='serif', fontstyle='italic')
    plt.ylabel('Cumulative Return', fontsize=12, fontfamily='serif', fontstyle='italic')

    plt.show()