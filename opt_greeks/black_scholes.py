
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
    def __init__(self, args, volatility = None, callPrice = None, putPrice = None):
        """
        args = [S, K, r, T]
        S = # Underlying asset price
        K = # Strike price
        r = # Risk-free rate
        T = # Time to maturity
        sigma = Volatility
        """
        self.underlyingPrice, self.strikePrice, self.interestRate, self.daysToExpiration = args
        self.volatility = volatility
        self.args = args

        for i in ['callPrice', 'putPrice', 'callDelta', 'putDelta', 
                'callDelta2', 'putDelta2', 'callTheta', 'putTheta', 
                'callRhoD', 'putRhoD', 'callRhoF', 'callRhoF', 'vega', 
                'gamma', 'impliedVolatility', 'putCallParity']:
            self.__dict__[i] = None

        if not self.volatility:
            if callPrice:
                self.callPrice = callPrice
                # self.impliedVolatility = self.implied_volatility(self.bs, self.args, self.callPrice, True)
                self.impliedVolatility = self.implied_volatility(args = self.args, option_price = self.callPrice, is_call = True)

            if putPrice and not callPrice:
                self.putPrice = putPrice
                # self.impliedVolatility = self.implied_volatility(self.bs, self.args, self.putPrice, False)
                self.impliedVolatility = self.implied_volatility(args = self.args, option_price = self.putPrice, is_call = False)

            if callPrice and putPrice:
                self.callPrice = float(callPrice)
                self.putPrice = float(putPrice)
                self.putCallParity = self._parity()
        else:
            [self.callPrice, self.putPrice, self._a_, self._d1_, self._d2_] = self._price()
            [self.callDelta, self.putDelta] = self._delta()
            [self.callDelta2, self.putDelta2] = self._delta2()
            [self.callTheta, self.putTheta] = self._theta()
            [self.callRho, self.putRho] = self._rho()
            self.vega = self._vega()
            self.gamma = self._gamma()
            self.exerciceProbability = norm.cdf(self._d2_)
    '''
    def implied_volatility2(self, model, argss, target, is_call, max_iter=10000):
        low, high = 0.0, 500.0
        decimals = len(str(target).split('.')[1])
        def mid_calc_func(model, argss, target, is_call):
            nonlocal low, high
            mid = (low + high) / 2
            if mid < 0.00001 : mid = 0.00001        

            # price_call, price_put = model(tuple(np.array(args, dtype=np.float64)), mid)
            price_call, price_put = model(argss, mid)
            price = price_call if is_call else price_put

            if round(price, decimals) == target:
                return mid
            elif price > target:
                high = mid
            else:
                low = mid
        
        return pd.Series(Parallel(n_jobs=-1)(delayed(mid_calc_func)(model, argss, target, is_call) for _ in range(max_iter))).dropna().drop_duplicates().values[0]

    def implied_volatility3(self, model, args, C, is_call, tol=1e-6, sigma_low=0.0, sigma_high=500.0):
        """
        Calculate the implied volatility of an option using the Black-Scholes model
        """
        S = args[0]
        K = args[1]
        r = args[2] / 100
        T = args[3] / 365
        # sigma_guess = 
        # price_low = black_scholes_call(S, K, r, T, sigma_low)
        # price_high = black_scholes_call(S, K, r, T, sigma_high)
        decimals = len(str(C).split('.')[1])
        condition = sigma_high - sigma_low > tol
        sigma_guess = np.where(condition, (sigma_low + sigma_high) / 2, 0)
        call, put, _, _, _ = model(args, sigma_guess)
        price_guess = call if is_call else put

        sigma_low = np.where(price_guess < C, sigma_guess, sigma_low)
        # price_low = np.where(price_guess < C, price_guess, price_low)

        sigma_high = np.where(price_guess >= C, sigma_guess, sigma_high)
        # price_high = np.where(price_guess >= C, price_guess, price_high)

        condition = sigma_high - sigma_low > tol
        while np.any(condition):
            
            sigma_guess = np.where(condition, (sigma_low + sigma_high) / 2, sigma_guess)
            call, put, _, _, _ = model(args, sigma_guess)
            price_guess = call if is_call else put
            if round(price_guess, decimals) == C:
                break
            sigma_low = np.where(np.logical_and(condition, price_guess < C), sigma_guess, sigma_low)
            # price_low = np.where(np.logical_and(condition, price_guess < C), price_guess, price_low)

            sigma_high = np.where(np.logical_and(condition, price_guess >= C), sigma_guess, sigma_high)
            # price_high = np.where(np.logical_and(condition, price_guess >= C), price_guess, price_high)

            condition = sigma_high - sigma_low > tol

        return sigma_guess
    '''
    
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
        if self.volatility == 0 or self.daysToExpiration == 0:
            call = max(0.0, self.underlyingPrice - self.strikePrice)
            put = max(0.0, self.strikePrice - self.underlyingPrice)
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            # call = self.underlyingPrice * norm.cdf(self._d1_) - \
            #         self.strikePrice * e**(-self.interestRate * \
            #         self.daysToExpiration) * norm.cdf(self._d2_)
            # put = self.strikePrice * e**(-self.interestRate * \
            #         self.daysToExpiration) * norm.cdf(-self._d2_) - \
            #         self.underlyingPrice * norm.cdf(-self._d1_)
            call, put, a, d1, d2 = self.bs(self.args, self.volatility)
        return [call, put, a, d1, d2]

    def _delta(self):
        '''Returns the option delta: [Call delta, Put delta]'''
        if self.volatility == 0 or self.daysToExpiration == 0:
            call = 1.0 if self.underlyingPrice > self.strikePrice else 0.0
            put = -1.0 if self.underlyingPrice < self.strikePrice else 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            call = norm.cdf(self._d1_)
            put = -norm.cdf(-self._d1_)
        return [call, put]

    def _delta2(self):
        '''Returns the dual delta: [Call dual delta, Put dual delta]'''
        if self.volatility == 0 or self.daysToExpiration == 0:
            call = -1.0 if self.underlyingPrice > self.strikePrice else 0.0
            put = 1.0 if self.underlyingPrice < self.strikePrice else 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            _b_ = e**-(self.interestRate * self.daysToExpiration)
            call = -norm.cdf(self._d2_) * _b_
            put = norm.cdf(-self._d2_) * _b_
        return [call, put]

    def _vega(self):
        '''Returns the option vega'''
        if self.volatility == 0 or self.daysToExpiration == 0:
            return 0.0
        if self.strikePrice == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        else:
            return self.underlyingPrice * norm.pdf(self._d1_) * \
                    self.daysToExpiration**0.5 / 100

    def _theta(self):
        '''Returns the option theta: [Call theta, Put theta]'''
        _b_ = e**-(self.interestRate * self.daysToExpiration)
        call = -self.underlyingPrice * norm.pdf(self._d1_) * self.volatility / \
                (2 * self.daysToExpiration**0.5) - self.interestRate * \
                self.strikePrice * _b_ * norm.cdf(self._d2_)
        put = -self.underlyingPrice * norm.pdf(self._d1_) * self.volatility / \
                (2 * self.daysToExpiration**0.5) + self.interestRate * \
                self.strikePrice * _b_ * norm.cdf(-self._d2_)
        return [call / 365, put / 365]

    def _rho(self):
        '''Returns the option rho: [Call rho, Put rho]'''
        _b_ = e**-(self.interestRate * self.daysToExpiration)
        call = self.strikePrice * self.daysToExpiration * _b_ * \
                norm.cdf(self._d2_) / 100
        put = -self.strikePrice * self.daysToExpiration * _b_ * \
                norm.cdf(-self._d2_) / 100
        return [call, put]

    def _gamma(self):
        '''Returns the option gamma'''
        return norm.pdf(self._d1_) / (self.underlyingPrice * self._a_)

    def _parity(self):
        '''Put-Call Parity'''
        return self.callPrice - self.putPrice - self.underlyingPrice + \
                (self.strikePrice / \
                ((1 + self.interestRate)**self.daysToExpiration))

    def bs(self, args, sigma):
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