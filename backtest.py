from typing import *
import numpy as np
import pandas as pd
import datetime

pd.options.mode.chained_assignment = None


class Backtest:
    def __init__(
            self,
            price: pd.Series,
            fees: float
    ) -> None:
        """
        Base class for backtesting price trend trading strategies
        """

        # save arguments in class scope
        self.fees = fees
        self.len = len(price)
        self.ix = np.arange(self.len)

        self.df = price.to_frame()
        self.df.columns = ['Price']

        # Specify trade signals
        self.create_signals()

        # Get positions
        self.df['Positions'] = np.full(len(self.df), np.nan)
        self.df['Positions'][self.df['Sig_L']] = 1
        self.df['Positions'][self.df['Sig_N']] = 0
        self.df['Positions'][self.df['Sig_S']] = -1
        if self.df['Positions'][0] == np.nan:
            self.df['Positions'][0] = 0
        self.df['Positions'].ffill(inplace=True)

        self.df['Trades'] = self.df['Positions'].diff().abs().astype(np.bool)

        # Calculate returns
        self.df['Returns Hold'] = price.pct_change()
        self.df['Returns Gross'] = self.df['Returns Hold'] * self.df['Positions']
        self.df['Returns Net'] = \
            self.df['Returns Gross'] - self.fees * self.df['Trades'] * (1 - self.df['Returns Gross'])

        self.df['Cum Returns Hold'] = (1 + self.df['Returns Hold']).cumprod() - 1
        self.df['Cum Returns Gross'] = (1 + self.df['Returns Gross']).cumprod() - 1
        self.df['Cum Returns Net'] = (1 + self.df['Returns Net']).cumprod() - 1

    def create_signals(self) -> None:
        """
        Specify signals for long, short and neutral
        """

        self.df['Sig_L'] = np.full(len(self.df), False)
        self.df['Sig_L'].iloc[0] = True
        self.df['Sig_S'] = np.full(len(self.df), False)
        self.df['Sig_N'] = np.full(len(self.df), False)

    def cum_returns(self) -> pd.DataFrame:
        """
        Return cumulative returns of strategy and hold
        """

        return self.df[['Cum Returns Net', 'Cum Returns Gross', 'Cum Returns Hold']].iloc[-1]

    def mean_returns(self) -> pd.DataFrame:
        """
        Return mean returns
        """

        return self.df[['Returns Net', 'Returns Gross', 'Returns Hold']].mean()

    def num_trades(self) -> float:
        """
        Return number of trades executed
        """

        return (self.df['Trades'] > 0).sum()


class TechAnalysis:
    @staticmethod
    def moving_avg(
            x: pd.Series,
            window: int,
            ewm: bool = False,
            groupfreq: AnyStr = ''
    ) -> np.ndarray:
        """
        Calculate moving averages
        """

        if groupfreq:
            x = x.groupby(pd.Grouper(freq=groupfreq))
            if ewm:
                return x.apply(lambda g: g.ewm(span=window).mean()).values
        if ewm:
            return x.ewm(span=window).mean().values
        return x.rolling(window=window).mean().values

    @staticmethod
    def moving_std(
            x: pd.Series,
            window: int,
            ewm: bool = False,
            groupfreq: AnyStr = ''
    ) -> np.ndarray:
        """
        Calculate moving standard deviation
        """

        if groupfreq:
            x = x.groupby(pd.Grouper(freq=groupfreq))
            if ewm:
                return x.apply(lambda g: g.ewm(span=window).std()).values
        if ewm:
            return x.ewm(span=window).std().values
        return x.rolling(window=window).std().values

    @staticmethod
    def rolling(
            x: pd.Series,
            window: int,
            func: Callable,
            groupfreq: AnyStr = ''
    ) -> np.ndarray:
        """
        Apply functions over rolling window
        """

        if groupfreq:
            x = x.groupby(pd.Grouper(freq=groupfreq))
        return x.rolling(window=window).apply(func).values

    @staticmethod
    def trading_hours(
            x: pd.core.indexes.datetimes.DatetimeIndex,
            t0_margin: datetime.timedelta = datetime.timedelta(0),
            t1_margin: datetime.timedelta = datetime.timedelta(0),
    ) -> np.ndarray:
        """
        Return boolean mask for trading hours
        """

        return x.to_series().groupby(pd.Grouper(freq='D'))\
            .apply(lambda t: (t.min() + t0_margin <= t) & (t <= t.max() - t1_margin))

    @staticmethod
    def mao(
            x: pd.Series,
            window: int,
            s_window: int,
            ewm: bool,
            groupfreq: AnyStr = ''
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Calculate moving average oscillator.
        """

        mas = TechAnalysis.moving_avg(x, s_window, ewm, groupfreq)
        mal = TechAnalysis.moving_avg(x, window, ewm, groupfreq)
        return mas, mal, (mas - mal) / mal * 100

    @staticmethod
    def rsin(
            x: pd.Series,
            window: int,
            ewm: bool,
            groupfreq: AnyStr = ''
    ) -> np.array:
        """
        Calculate normalized relative strength index.
        """

        ma_u = TechAnalysis.moving_avg(np.maximum(x.diff(), 0), window, ewm, groupfreq)
        ma_l = TechAnalysis.moving_avg(-np.minimum(x.diff(), 0), window, ewm, groupfreq)
        return (ma_u - ma_l) / (ma_u + ma_l)


class SG1(Backtest):
    def __init__(
            self,
            price: pd.Series,
            fees: float,
            oscillator: AnyStr,
            window: int,
            s_window: Optional[int] = None,
            ewm: bool = False,
    ) -> None:
        """
        Trend following strategy SG1 from doi:10.1016/j.rfe.2008.10.001
        """

        assert (oscillator in ('RSIN', 'MAO') if isinstance(oscillator, str) else False)
        assert (s_window is not None if oscillator == 'MAO' else True)

        self.oscillator = oscillator
        self.window, self.s_window = window, s_window
        self.ewm = ewm
        Backtest.__init__(self, price=price, fees=fees)

    def create_signals(self) -> None:
        if self.oscillator == 'MAO':
            self.df['MAS'], self.df['MAL'], self.df['Oscillator'] = \
                TechAnalysis.mao(self.df['Price'], self.window, self.s_window, self.ewm, groupfreq='D')
        else:  # self.oscillator == 'RSIN'
            self.df['Oscillator'] = TechAnalysis.rsin(self.df['Price'], self.window, self.ewm, groupfreq='D')

        hours = TechAnalysis.trading_hours(
            self.df.index,
            t0_margin=self.df.index[self.window] - self.df.index[0],
            t1_margin=datetime.timedelta(minutes=3)
        )
        osc_cross = np.sign(self.df['Oscillator']).diff()
        self.df['Sig_L'] = (osc_cross > 0) & hours
        self.df['Sig_S'] = osc_cross < 0 & hours
        self.df['Sig_N'] = ~hours


class SG2(Backtest):
    def __init__(
            self,
            price: pd.Series,
            fees: float,
            oscillator: AnyStr,
            window: int,
            s_window: Optional[int] = None,
            band_type: AnyStr = 'offset',
            band_alpha: float = 1.,
            ewm: bool = False,
    ) -> None:
        """
        Trend following strategy SG2 from doi:10.1016/j.rfe.2008.10.001
        """

        assert (oscillator in ('RSIN', 'MAO') if isinstance(oscillator, str) else False)
        assert (s_window is not None if oscillator == 'MAO' else True)
        assert (band_type in ('offset', 'bollinger') if isinstance(band_type, str) else False)

        self.oscillator = oscillator
        self.window, self.s_window = window, s_window
        self.band_type, self.band_alpha = band_type, band_alpha
        self.ewm = ewm
        Backtest.__init__(self, price=price, fees=fees)

    def create_signals(self) -> None:
        if self.oscillator == 'MAO':
            self.df['MAS'], self.df['MAL'], self.df['Oscillator'] = \
                TechAnalysis.mao(self.df['Price'], self.window, self.s_window, self.ewm, groupfreq='D')
        else:  # self.oscillator == 'RSIN'
            self.df['Oscillator'] = TechAnalysis.rsin(self.df['Price'], self.window, self.ewm, groupfreq='D')

        if self.band_type == 'offset':
            self.df['UB'] = self.band_alpha
        else:  # self.bands == 'bollinger'
            self.df['UB'] = self.band_alpha * TechAnalysis.moving_std(self.df['Oscillator'], self.window, ewm=self.ewm)
        self.df['LB'] = -self.df['UB']

        hours = TechAnalysis.trading_hours(
            self.df.index,
            t0_margin=self.df.index[self.window] - self.df.index[0],
            t1_margin=datetime.timedelta(minutes=3)
        )
        osc_cross = np.sign(self.df['Oscillator']).diff()
        ub_cross = np.sign(self.df['Oscillator'] - self.df['UB']).diff()
        lb_cross = np.sign(self.df['Oscillator'] - self.df['LB']).diff()
        self.df['Sig_L'] = (ub_cross > 0) & hours
        self.df['Sig_S'] = (lb_cross < 0) & hours
        self.df['Sig_N'] = (osc_cross != 0) & ~self.df['Sig_L'] & ~self.df['Sig_S'] | ~hours


class SG3(Backtest):
    def __init__(
            self,
            price: pd.Series,
            fees: float,
            oscillator: AnyStr,
            window: int,
            s_window: Optional[int] = None,
            band_type: AnyStr = 'offset',
            band_alpha: float = 1.,
            ewm: bool = False,
    ) -> None:
        """
        Trend following strategy SG3 from doi:10.1016/j.rfe.2008.10.001
        """

        assert (oscillator in ('RSIN', 'MAO') if isinstance(oscillator, str) else False)
        assert (s_window is not None if oscillator == 'MAO' else True)
        assert (band_type in ('offset', 'bollinger') if isinstance(band_type, str) else False)

        self.oscillator = oscillator
        self.window, self.s_window = window, s_window
        self.band_type, self.band_alpha = band_type, band_alpha
        self.ewm = ewm
        Backtest.__init__(self, price=price, fees=fees)

    def create_signals(self) -> None:
        if self.oscillator == 'MAO':
            self.df['MAS'], self.df['MAL'], self.df['Oscillator'] = \
                TechAnalysis.mao(self.df['Price'], self.window, self.s_window, self.ewm, groupfreq='D')
        else:  # self.oscillator == 'RSIN'
            self.df['Oscillator'] = TechAnalysis.rsin(self.df['Price'], self.window, self.ewm, groupfreq='D')

        if self.band_type == 'offset':
            self.df['UB'] = self.band_alpha
        else:  # self.bands == 'bollinger'
            self.df['UB'] = self.band_alpha * TechAnalysis.moving_std(self.df['Oscillator'], self.window, ewm=self.ewm)
        self.df['LB'] = -self.df['UB']

        hours = TechAnalysis.trading_hours(
            self.df.index,
            t0_margin=self.df.index[self.window] - self.df.index[0],
            t1_margin=datetime.timedelta(minutes=3)
        )
        ub_cross = np.sign(self.df['Oscillator'] - self.df['UB']).diff()
        lb_cross = np.sign(self.df['Oscillator'] - self.df['LB']).diff()
        self.df['Sig_L'] = (ub_cross > 0) & hours
        self.df['Sig_S'] = (lb_cross < 0) & hours
        self.df['Sig_N'] = ((ub_cross < 0) | (lb_cross > 0)) & ~self.df['Sig_L'] & ~self.df['Sig_S'] | ~hours


class SG4(Backtest):
    def __init__(
            self,
            price: pd.Series,
            fees: float,
            oscillator: AnyStr,
            window: int,
            s_window: Optional[int] = None,
            band_type: AnyStr = 'offset',
            band_alpha: float = 1.,
            ewm: bool = False,
    ) -> None:
        """
        Contrarian strategy SG5 from doi:10.1016/j.rfe.2008.10.001
        """

        assert (oscillator in ('RSIN', 'MAO') if isinstance(oscillator, str) else False)
        assert (s_window is not None if oscillator == 'MAO' else True)
        assert (band_type in ('offset', 'bollinger') if isinstance(band_type, str) else False)

        self.oscillator = oscillator
        self.window, self.s_window = window, s_window
        self.band_type, self.band_alpha = band_type, band_alpha
        self.ewm = ewm
        Backtest.__init__(self, price=price, fees=fees)

    def create_signals(self) -> None:
        if self.oscillator == 'MAO':
            self.df['MAS'], self.df['MAL'], self.df['Oscillator'] = \
                TechAnalysis.mao(self.df['Price'], self.window, self.s_window, self.ewm, groupfreq='D')
        else:  # self.oscillator == 'RSIN'
            self.df['Oscillator'] = TechAnalysis.rsin(self.df['Price'], self.window, self.ewm, groupfreq='D')

        if self.band_type == 'offset':
            self.df['UB'] = self.band_alpha
        else:  # self.bands == 'bollinger'
            self.df['UB'] = self.band_alpha * TechAnalysis.moving_std(self.df['Oscillator'], self.window, ewm=self.ewm)
        self.df['LB'] = -self.df['UB']

        hours = TechAnalysis.trading_hours(
            self.df.index,
            t0_margin=self.df.index[self.window] - self.df.index[0],
            t1_margin=datetime.timedelta(minutes=3)
        )
        ub_cross = np.sign(self.df['Oscillator'] - self.df['UB']).diff()
        lb_cross = np.sign(self.df['Oscillator'] - self.df['LB']).diff()
        self.df['Sig_L'] = (lb_cross > 0) & hours
        self.df['Sig_S'] = (ub_cross < 0) & hours
        self.df['Sig_N'] = ~hours


class SG5(Backtest):
    def __init__(
            self,
            price: pd.Series,
            fees: float,
            oscillator: AnyStr,
            window: int,
            s_window: Optional[int] = None,
            band_type: AnyStr = 'offset',
            band_alpha: float = 1.,
            ewm: bool = False,
    ) -> None:
        """
        Contrarian strategy SG5 from doi:10.1016/j.rfe.2008.10.001
        """

        assert (oscillator in ('RSIN', 'MAO') if isinstance(oscillator, str) else False)
        assert (s_window is not None if oscillator == 'MAO' else True)
        assert (band_type in ('offset', 'bollinger') if isinstance(band_type, str) else False)

        self.oscillator = oscillator
        self.window, self.s_window = window, s_window
        self.band_type, self.band_alpha = band_type, band_alpha
        self.ewm = ewm
        Backtest.__init__(self, price=price, fees=fees)

    def create_signals(self) -> None:
        if self.oscillator == 'MAO':
            self.df['MAS'], self.df['MAL'], self.df['Oscillator'] = \
                TechAnalysis.mao(self.df['Price'], self.window, self.s_window, self.ewm, groupfreq='D')
        else:  # self.oscillator == 'RSIN'
            self.df['Oscillator'] = TechAnalysis.rsin(self.df['Price'], self.window, self.ewm, groupfreq='D')

        if self.band_type == 'offset':
            self.df['UB'] = self.band_alpha
        else:  # self.bands == 'bollinger'
            self.df['UB'] = self.band_alpha * TechAnalysis.moving_std(self.df['Oscillator'], self.window, ewm=self.ewm)
        self.df['LB'] = -self.df['UB']

        hours = TechAnalysis.trading_hours(
            self.df.index,
            t0_margin=self.df.index[self.window] - self.df.index[0],
            t1_margin=datetime.timedelta(minutes=3)
        )
        osc_cross = np.sign(self.df['Oscillator']).diff()
        ub_cross = np.sign(self.df['Oscillator'] - self.df['UB']).diff()
        lb_cross = np.sign(self.df['Oscillator'] - self.df['LB']).diff()
        self.df['Sig_L'] = (osc_cross > 0) & hours
        self.df['Sig_S'] = (osc_cross < 0) & hours
        self.df['Sig_N'] = ((ub_cross < 0) | (lb_cross > 0)) & ~self.df['Sig_L'] & ~self.df['Sig_S'] | ~hours


class SG6(Backtest):
    def __init__(
            self,
            price: pd.Series,
            fees: float,
            oscillator: AnyStr,
            window: int,
            s_window: Optional[int] = None,
            band_type: AnyStr = 'offset',
            band_alpha: float = 1.,
            band_alpha2: float = .5,
            ewm: bool = False,
    ) -> None:
        """
        Contrarian strategy SG6 from doi:10.1016/j.rfe.2008.10.001
        """

        assert (oscillator in ('RSIN', 'MAO') if isinstance(oscillator, str) else False)
        assert (s_window is not None if oscillator == 'MAO' else True)
        assert (band_type in ('offset', 'bollinger') if isinstance(band_type, str) else False)

        self.oscillator = oscillator
        self.window, self.s_window = window, s_window
        self.band_type, self.band_alpha, self.band_alpha2 = band_type, band_alpha, band_alpha2
        self.ewm = ewm
        Backtest.__init__(self, price=price, fees=fees)

    def create_signals(self) -> None:
        if self.oscillator == 'MAO':
            self.df['MAS'], self.df['MAL'], self.df['Oscillator'] = \
                TechAnalysis.mao(self.df['Price'], self.window, self.s_window, self.ewm, groupfreq='D')
        else:  # self.oscillator == 'RSIN'
            self.df['Oscillator'] = TechAnalysis.rsin(self.df['Price'], self.window, self.ewm, groupfreq='D')

        if self.band_type == 'offset':
            self.df['UB'], self.df['UB2'] = self.band_alpha, self.band_alpha2
        else:  # self.bands == 'bollinger'
            std = TechAnalysis.moving_std(self.df['Oscillator'], self.window, ewm=self.ewm)
            self.df['UB'], self.df['UB2'] = self.band_alpha * std, self.band_alpha2 * std
        self.df['LB'], self.df['LB2'] = -self.df['UB'], -self.df['UB2']

        hours = TechAnalysis.trading_hours(
            self.df.index,
            t0_margin=self.df.index[self.window] - self.df.index[0],
            t1_margin=datetime.timedelta(minutes=3)
        )
        ub_cross = np.sign(self.df['Oscillator'] - self.df['UB']).diff()
        lb_cross = np.sign(self.df['Oscillator'] - self.df['LB']).diff()
        ub2_cross = np.sign(self.df['Oscillator'] - self.df['UB2']).diff()
        lb2_cross = np.sign(self.df['Oscillator'] - self.df['LB2']).diff()
        self.df['Sig_L'] = (lb2_cross > 0) & hours
        self.df['Sig_S'] = (ub2_cross < 0) & hours
        self.df['Sig_N'] = ((ub_cross < 0) | (lb_cross > 0)) & ~self.df['Sig_L'] & ~self.df['Sig_S'] | ~hours
