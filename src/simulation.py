import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from data import get_market_data

DEFAULT_DAYS = 30
DEFAULT_SIMULATIONS = 10000
TRADING_DAYS = 252


def monte_carlo_simulation(initial_price, expected_return, volatility, days, simulations):
    dt = 1 / TRADING_DAYS
    random_shocks = np.random.normal(0, 1, size=(days, simulations))
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = initial_price

    for day in range(1, days + 1):
        price_paths[day] = price_paths[day - 1] * np.exp(
            (expected_return - 0.5 * volatility ** 2) * dt
            + volatility * np.sqrt(dt) * random_shocks[day - 1]
        )

    return price_paths


def prompt_ticker():
    while True:
        ticker = input("Enter the stock ticker symbol: ").strip().upper()
        if ticker:
            return ticker
        print("Ticker symbol not recognized, please enter valid input.")


def prompt_positive_int(prompt, default):
    while True:
        raw = input(prompt).strip()
        if raw == "":
            print(f"Invalid input. Defaulting to {default}.")
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Invalid character. Please enter a number.")
            continue
        if value <= 0:
            print(f"Invalid input. Defaulting to {default}.")
            return default
        return value


def main():
    ticker = prompt_ticker()
    days = prompt_positive_int("Enter the number of days to simulate: ", DEFAULT_DAYS)
    simulations = prompt_positive_int("Enter the number of simulations to run: ", DEFAULT_SIMULATIONS)

    data, initial_price, expected_return, volatility = get_market_data(ticker)

    price_paths = monte_carlo_simulation(initial_price, expected_return, volatility, days, simulations)
    final_prices = price_paths[-1]

    probability_of_profit = np.mean(final_prices > initial_price)
    var_price = np.percentile(final_prices, 5)
    var_95 = (var_price - initial_price) / initial_price
    tail_prices = final_prices[final_prices <= var_price]
    cvar_price = np.mean(tail_prices)
    cvar_95 = (cvar_price - initial_price) / initial_price
    expected_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    lower_bound = np.percentile(final_prices, 2.5)
    upper_bound = np.percentile(final_prices, 97.5)

    company = yf.Ticker(ticker).info.get("shortName")

    print()
    print("Monte Carlo Forecast")
    print("-" * 45)
    print(f"{company} ({ticker})")
    print()
    print(f"Current price:                      ${initial_price:.2f}")
    print()
    print(f"Time horizon:                       {days} days")
    print(f"Simulations:                        {simulations:,}")
    print()
    print(f"Expected price:                     ${expected_price:.2f}")
    print(f"Median price:                       ${median_price:.2f}")
    print()
    print("95% Confidence Interval:")
    print(f"Lower Bound:                        ${lower_bound:.2f}")
    print(f"Upper Bound:                        ${upper_bound:.2f}")
    print()
    print(f"Probability of Profit:              {probability_of_profit:.2%}")
    print()
    print(f"Value at Risk (5%):                 {var_95:.2%}")
    print(f"Conditional Value at Risk (5%):     {cvar_95:.2%}")

    plot_results(price_paths, final_prices, initial_price, median_price, lower_bound, upper_bound)


def plot_results(price_paths, final_prices, initial_price, median_price, lower_bound, upper_bound):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(price_paths[:, :100], linewidth=0.7, alpha=0.5)
    ax1.axhline(initial_price, linestyle="--", color="black", linewidth=1, label="Initial Price")
    ax1.set_title("Monte Carlo Simulated Price Paths")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(alpha=0.2)

    sns.histplot(final_prices, bins=100, kde=True, color="grey", edgecolor=None, ax=ax2)

    if ax2.lines:
        kde_line = ax2.lines[0]
        kde_x = kde_line.get_xdata()
        kde_y = kde_line.get_ydata()
        y_lower = np.interp(lower_bound, kde_x, kde_y)
        y_upper = np.interp(upper_bound, kde_x, kde_y)
    else:
        kde_x, kde_y, y_lower, y_upper = np.array([]), np.array([]), 0, 0

    ax2.vlines(x=lower_bound, ymin=0, ymax=y_lower, color="grey", linestyle="-",
               linewidth=1.5, label=f"2.5%: ${lower_bound:.2f}")
    ax2.vlines(x=upper_bound, ymin=0, ymax=y_upper, color="grey", linestyle="-",
               linewidth=1.5, label=f"97.5%: ${upper_bound:.2f}")

    shade_mask = (kde_x >= lower_bound) & (kde_x <= upper_bound)
    ax2.fill_between(kde_x[shade_mask], 0, kde_y[shade_mask], color="skyblue",
                      alpha=0.4, label="95% Confidence Interval")

    ax2.axvline(x=initial_price, color="black", linestyle="-", linewidth=1, label=f"Initial: ${initial_price:.2f}")
    ax2.axvline(x=median_price, color="black", linestyle=":", linewidth=1, label=f"Median: ${median_price:.2f}")

    ax2.set_title("Terminal Price Distribution")
    ax2.set_xlabel("Price ($)")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()