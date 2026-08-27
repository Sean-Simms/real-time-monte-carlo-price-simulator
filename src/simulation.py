import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_simulation(
    initial_price,
    expected_return,
    volatility,
    days,
    simulations
):
    """
    Run a Monte Carlo simulation using Geometric Brownian Motion.

    Parameters
    ----------
    initial_price : float
        Current price of the asset.

    expected_return : float
        Expected annualised return as a decimal.
        Example: 0.08 = 8%.

    volatility : float
        Annualised volatility as a decimal.
        Example: 0.25 = 25%.

    days : int
        Number of trading days to simulate.

    simulations : int
        Number of Monte Carlo simulations to run.

    Returns
    -------
    np.ndarray
        Matrix containing simulated price paths.
    """

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


initial_price = 100
expected_return = 0.08
volatility = 0.25
days = 30
simulations = 10_000


price_paths = monte_carlo_simulation(initial_price,expected_return,volatility,days,simulations)
final_prices = price_paths[-1]
expected_price = np.mean(final_prices)
median_price = np.median(final_prices)
percentile_5 = np.percentile(final_prices, 5)
percentile_95 = np.percentile(final_prices, 95)


print("Monte Carlo Price Simulation")
print("-" * 35)
print(f"Initial price:     ${initial_price:.2f}")
print(f"Expected return:   {expected_return:.2%}")
print(f"Volatility:        {volatility:.2%}")
print(f"Time horizon:      {days} days")
print(f"Simulations:       {simulations:,}")
print()
print("Simulation Results")
print("-" * 35)
print(f"Expected price:    ${expected_price:.2f}")
print(f"Median price:      ${median_price:.2f}")
print(f"5th percentile:    ${percentile_5:.2f}")
print(f"95th percentile:   ${percentile_95:.2f}")


plt.figure(figsize=(12, 6))
plt.plot(price_paths[:, :100], linewidth=0.7, alpha=0.5)
plt.axhline(initial_price,linestyle="--",label="Initial Price")
plt.title("Monte Carlo Simulated Price Paths")
plt.xlabel("Trading Days")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()