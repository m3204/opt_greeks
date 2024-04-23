
import numpy as np
from math import log, e
from scipy.stats import norm
from scipy.interpolate import interp1d

'''
    # USING Markov chain 
    def calculate_option_price(self, S, K, r, T, sigma, is_call):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        # d1 = (np.log(S / K) + (r + sigma ** 2) * T) / (sigma * np.sqrt(T))

        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return option_price


    def implied_volatility(self, args, option_price, is_call, tolerance=1e-6, max_iter=10000):
        iv = 0.5  # Initial guess for IV
        S = float(args[0])
        K = float(args[1])
        r = float(args[2] / 100)
        T = float(args[3] / 365)

        for i in range(max_iter):
            option_price_calculated = self.calculate_option_price(S, K, r, T, iv, is_call)
            vega = S * np.sqrt(T) * norm.pdf((np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T)))
            # vega = S * np.sqrt(T) * norm.pdf((np.log(S / K) + (r + iv ** 2) * T) / (iv * np.sqrt(T)))

            diff = option_price_calculated - option_price
            print(option_price_calculated)
            if abs(diff) < tolerance:
                return iv
            iv -= diff / vega

        return iv
'''

class BlackScholes:
    def __init__(self, args, volatility = None, call_price = None, put_price = None):
        """
        args = [S, K, r, T]
        S = # Underlying asset price
        K = # Strike price
        r = # Risk-free rate
        T = # Time to maturity
        sigma = Volatility
        """
        self.underlying_price, self.strike_price, self.interest_rate, self.days_to_expiration = args
        self.volatility = volatility
        self.args = args

        for i in ['call_price', 'put_price', 'call_delta', 'put_delta', 
                'call_delta2', 'put_delta2', 'call_theta', 'put_theta', 
                'call_rhod', 'put_rhod', 'call_rhof', 'call_rhof', 'vega', 
                'gamma', 'implied_volatility', 'put_call_parity']:
            self.__dict__[i] = None

        if not self.volatility:
            if call_price:
                self.call_price = call_price
                # self.implied_volatility = self.implied_volatility(self.bs, self.args, self.call_price, True)
                self.implied_volatility = self.implied_volatility(args = self.args, option_price = self.call_price, is_call = True)

            if put_price and not call_price:
                self.put_price = put_price
                # self.implied_volatility = self.implied_volatility(self.bs, self.args, self.put_price, False)
                self.implied_volatility = self.implied_volatility(args = self.args, option_price = self.put_price, is_call = False)

            if call_price and put_price:
                self.call_price = float(call_price)
                self.put_price = float(put_price)
                self.put_call_parity = self._parity()
        else:
            [self.call_price, self.put_price, self._a_, self._d1_, self._d2_] = self._price()
            [self.call_delta, self.put_delta] = self._delta()
            [self.call_delta2, self.put_delta2] = self._delta2()
            [self.call_theta, self.put_theta] = self._theta()
            [self.callRho, self.putRho] = self._rho()
            self.vega = self._vega()
            self.gamma = self._gamma()
            self.exerciceProbability = norm.cdf(self._d2_)
    
    # FFT
    def price_options_fft(self, S, K, T, r, volatility_grid, is_call):
        # Implement FFT pricing for each volatility in the grid
        d1 = (np.log(S / K) + (r + 0.5 * volatility_grid ** 2) * T) / (volatility_grid * np.sqrt(T))
        # d1 = (np.log(S / K) + (r + volatility_grid ** 2) * T) / (volatility_grid * np.sqrt(T))

        d2 = d1 - volatility_grid * np.sqrt(T)

        # if option_type == 'call':
        if is_call:
            option_prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        # elif option_type == 'put':
        elif not is_call:
            option_prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return option_prices

    # Function to calculate implied volatility from option prices using FFT
    def implied_volatility(self, args, option_price, is_call):
        # Price options for the given volatility grid
        S = float(args[0])
        K = float(args[1])
        r = float(args[2] / 100)
        T = float(args[3] / 365)
        volatility_grid = np.linspace(0.001, 5, 10000)
        # volatility_grid = np.linspace(0.01, 5, 100)
        option_prices = self.price_options_fft(S, K, T, r, volatility_grid, is_call)
        # Interpolate option prices to find implied volatility
        interp_func = interp1d(option_prices, volatility_grid, kind='linear')
        # interp_func = interp1d(option_prices, volatility_grid, kind='next')
        # print('nearest')
        try:
            implied_volatility = interp_func(option_price)
            implied_volatility = float(implied_volatility) * 100
        except Exception as e:
            print('No IV', e)
            implied_volatility = 1e-5
        return implied_volatility
    
    def _price(self):
        '''Returns the option price: [Call price, Put price]'''
        if self.volatility == 0 or self.days_to_expiration == 0:
            call = max(0.0, self.underlying_price - self.strike_price)
            put = max(0.0, self.strike_price - self.underlying_price)
        if self.strike_price == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            call, put, a, d1, d2 = self.bs(self.args, self.volatility)
        return [call, put, a, d1, d2]

    def _delta(self):
        '''Returns the option delta: [Call delta, Put delta]'''
        if self.volatility == 0 or self.days_to_expiration == 0:
            call = 1.0 if self.underlying_price > self.strike_price else 0.0
            put = -1.0 if self.underlying_price < self.strike_price else 0.0
        if self.strike_price == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            call = norm.cdf(self._d1_)
            put = -norm.cdf(-self._d1_)
        return [call, put]

    def _delta2(self):
        '''Returns the dual delta: [Call dual delta, Put dual delta]'''
        if self.volatility == 0 or self.days_to_expiration == 0:
            call = -1.0 if self.underlying_price > self.strike_price else 0.0
            put = 1.0 if self.underlying_price < self.strike_price else 0.0
        if self.strike_price == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            _b_ = e**-(self.interest_rate * self.days_to_expiration)
            call = -norm.cdf(self._d2_) * _b_
            put = norm.cdf(-self._d2_) * _b_
        return [call, put]

    def _vega(self):
        '''Returns the option vega'''
        if self.volatility == 0 or self.days_to_expiration == 0:
            return 0.0
        if self.strike_price == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            return self.underlying_price * norm.pdf(self._d1_) * \
                    self.days_to_expiration**0.5 / 100

    def _theta(self):
        '''Returns the option theta: [Call theta, Put theta]'''
        _b_ = e**-(self.interest_rate * self.days_to_expiration)
        call = -self.underlying_price * norm.pdf(self._d1_) * self.volatility / \
                (2 * self.days_to_expiration**0.5) - self.interest_rate * \
                self.strike_price * _b_ * norm.cdf(self._d2_)
        put = -self.underlying_price * norm.pdf(self._d1_) * self.volatility / \
                (2 * self.days_to_expiration**0.5) + self.interest_rate * \
                self.strike_price * _b_ * norm.cdf(-self._d2_)
        return [call / 365, put / 365]

    def _rho(self):
        '''Returns the option rho: [Call rho, Put rho]'''
        _b_ = e**-(self.interest_rate * self.days_to_expiration)
        call = self.strike_price * self.days_to_expiration * _b_ * \
                norm.cdf(self._d2_) / 100
        put = -self.strike_price * self.days_to_expiration * _b_ * \
                norm.cdf(-self._d2_) / 100
        return [call, put]

    def _gamma(self):
        '''Returns the option gamma'''
        return norm.pdf(self._d1_) / (self.underlying_price * self._a_)

    def _parity(self):
        '''Put-Call Parity'''
        return self.call_price - self.put_price - self.underlying_price + \
                (self.strike_price / \
                ((1 + self.interest_rate)**self.days_to_expiration))

    def bs(self, args, sigma):
        '''call put value'''
        S, K, r, T = args
        r = float(r) / 100
        T = float(T) / 365
        sigma = float(sigma) / 100
        a = sigma * (T ** 0.5)
        d1 = (log(S / K) + (r + (sigma**2) / 2) * T) / a
        d2 = d1 - a
        call = S * norm.cdf(d1) - K * e ** (-r * T) * norm.cdf(d2)
        put = K * e ** (-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return call, put, a, d1, d2