import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def monte_carlo_simulation(initial_price,expected_return,volatility,days,simulations):
    
    trading_days = 252
    dt = 1 / trading_days
    random_shocks = np.random.normal(0,1,size=(days, simulations))
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = initial_price

    for day in range(1, days + 1):
        price_paths[day] = price_paths[day - 1] * np.exp(
            (
                expected_return
                - 0.5 * volatility ** 2
            ) * dt
            + volatility * np.sqrt(dt) * random_shocks[day - 1]
        )
    return price_paths


from data import get_market_data
ticker = "AAPL"
data, initial_price, expected_return, volatility = get_market_data(ticker)
days = 30
simulations = 10_000


price_paths = monte_carlo_simulation(initial_price,expected_return,volatility,days,simulations)
final_prices = price_paths[-1]
expected_price = np.mean(final_prices)
median_price = np.median(final_prices)
percentile_5 = np.percentile(final_prices, 5)
percentile_95 = np.percentile(final_prices, 95)


print("Monte Carlo Price Simulation")
print("-" * 40)
print(f"Ticker:            {ticker}")
print(f"Current price:     ${initial_price:.2f}")
print(f"Expected return:   {expected_return:.2%}")
print(f"Volatility:        {volatility:.2%}")
print(f"Time horizon:      {days} days")
print(f"Simulations:       {simulations:,}")
print()
print("Simulation Results")
print("-" * 40)
print(f"Expected price:    ${expected_price:.2f}")
print(f"Median price:      ${median_price:.2f}")
print(f"5th percentile:    ${percentile_5:.2f}")
print(f"95th percentile:   ${percentile_95:.2f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(price_paths[:, :100], linewidth=0.7, alpha=0.5)
ax1.axhline(initial_price,linestyle="--",color="black", linewidth=1, label="Initial Price")
ax1.set_title("Monte Carlo Simulated Price Paths")
ax1.set_xlabel("Trading Days")
ax1.set_ylabel("Price ($)")
ax1.legend()
ax1.grid(alpha=0.2)

lower_bound = np.percentile(final_prices, 2.5)
upper_bound = np.percentile(final_prices, 97.5)
sns.histplot(final_prices, bins=100, kde=True, color="grey", edgecolor=None, ax=ax2)
kde_line = ax2.lines[0]
kde_x = kde_line.get_xdata()
kde_y = kde_line.get_ydata()
y_lower = np.interp(lower_bound, kde_x, kde_y)
y_upper = np.interp(upper_bound, kde_x, kde_y)
ax2.vlines(x=lower_bound, ymin=0, ymax=y_lower, color="grey", linestyle="-", linewidth=1.5, label=f"2.5%: ${lower_bound:.2f}")
ax2.vlines(x=upper_bound, ymin=0, ymax=y_upper, color="grey", linestyle="-", linewidth=1.5, label=f"97.5%: ${upper_bound:.2f}")
shade_mask = (kde_x >= lower_bound) & (kde_x <= upper_bound)
ax2.fill_between(kde_x[shade_mask], 0, kde_y[shade_mask], color="skyblue", alpha=0.4, label="95% Confidence Interval")
y_initial = np.interp(initial_price, kde_x, kde_y)
y_median = np.interp(median_price, kde_x, kde_y)
ax2.axvline(x=initial_price, color="black", linestyle="-", linewidth=1, label=f"Initial: ${initial_price:.2f}")
ax2.axvline(x=median_price, color="black", linestyle=":", linewidth=1, label=f"Median: ${median_price:.2f}")
ax2.set_title("Terminal Price Distribution")
ax2.set_xlabel("Price ($)")
ax2.set_ylabel("Count")
ax2.legend()
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.show()