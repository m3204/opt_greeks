import pytest
from opt_greeks import BlackScholes

# Define fixture for initializing BlackScholes object with example arguments
@pytest.fixture
def black_scholes_object():
    args = [100, 100, 0.05, 30]  # Example arguments: [S, K, r, T]
    return BlackScholes(args)

# Test case for checking initialization of BlackScholes object
def test_black_scholes_initialization(black_scholes_object):
    assert isinstance(black_scholes_object, BlackScholes)

# Test case for checking implied volatility calculation
def test_implied_volatility_calculation(black_scholes_object):
    # Assuming a call option with a known price
    option_price = 5.0
    is_call = True
    # Calculate implied volatility
    implied_volatility2 = black_scholes_object.implied_volatility(black_scholes_object.args, option_price, is_call)
    # Assert that implied volatility is within a reasonable range
    assert 0 <= implied_volatility2 <= 200
