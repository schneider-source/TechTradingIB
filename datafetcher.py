from typing import *
import os
import json
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas_datareader as pdr
from iexfinance.stocks import get_historical_intraday


class DataFetcher:
    """
    Methods to fetch historical market data
    """

    @staticmethod
    def __apply(
            func: Callable,
            arg: Union[Any, Iterable[Any]],
            *args,
            **kwargs
    ) -> Union[Any, Iterable[Any]]:
        """
        Apply method to scalar or iterable
        """

        if np.isscalar(arg):
            return func(arg, *args, **kwargs)
        else:
            return [func(a, *args, **kwargs) for a in arg]

    @staticmethod
    def __to(
            data: pd.DataFrame,
            fname: AnyStr = ''
    ) -> Optional[pd.DataFrame]:
        """
        Return data or save to file
        """

        if fname:
            data.to_csv(fname)
        else:
            return data

    @staticmethod
    def __get_intraday_asset_data(
            symbol: Union[AnyStr, Iterable[AnyStr]],
            start: datetime,
            end: datetime
    ) -> pd.DataFrame:
        """
        Retrieve intraday historical market date from iexfinance.com
        """

        data = pd.DataFrame([])
        for i in tqdm(range((end - start).days + 1)):
            date = start + datetime.timedelta(days=i)
            df = get_historical_intraday(symbol, date, output_format='pandas').fillna(method='ffill')
            data = pd.concat([data, df], 0, sort=True)
        data.name = symbol
        return data

    @staticmethod
    def get_intraday_asset_data(
            symbol: Union[AnyStr, Iterable[AnyStr]],
            start: datetime,
            end: datetime,
            fname: AnyStr = ''
    ) -> pd.DataFrame:
        """
        Retrieve intraday historical market date from iexfinance.com
        """
        with open('iex.json') as f:
            os.environ['IEX_TOKEN'] = json.load(f)['TOKEN']

        data = DataFetcher.__apply(DataFetcher.__get_intraday_asset_data, symbol, start, end)
        return DataFetcher.__to(data, fname)

    @staticmethod
    def __get_daily_asset_data(
            symbol: Union[AnyStr, Iterable[AnyStr]],
            start: datetime,
            end: datetime
    ) -> pd.DataFrame:
        """
        Retrieve daily historical market data with pandas datareader from yahoo finance
        """

        data = pdr.DataReader(symbol, 'yahoo', start=start, end=end).fillna(method='ffill')
        data.name = symbol
        return data

    @staticmethod
    def get_daily_asset_data(
            symbol: Union[AnyStr, Iterable[AnyStr]],
            start: datetime,
            end: datetime,
            fname: AnyStr = ''
    ) -> pd.DataFrame:
        """
        Retrieve daily historical market data with pandas datareader from yahoo finance
        """

        data = DataFetcher.__apply(DataFetcher.__get_daily_asset_data, symbol, start, end)
        return DataFetcher.__to(data, fname)

    @staticmethod
    def from_csv(fname: AnyStr) -> pd.DataFrame:
        """
        Load market data from csv file
        """

        data = pd.read_csv(fname, index_col=0)
        data.index = pd.to_datetime(data.index)
        return data
