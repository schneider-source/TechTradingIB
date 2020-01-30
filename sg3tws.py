import time
import itertools
import threading
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from util import aprint, Path
from api_util import IBContract, IBOrder
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.ticktype import TickTypeEnum


class SG3TWS(EWrapper, EClient):
    def __init__(
            self,
            contract: IBContract,
            invest: float,
            s_window: int,
            l_window: int,
            update_interval: int = 15,
            band_offset: float = 0.1,
    ) -> None:
        assert (isinstance(contract, IBContract))
        assert (0 < invest if isinstance(invest, float) else False)
        assert (0 < s_window if isinstance(s_window, int) else False)
        assert (0 < l_window if isinstance(l_window, int) else False)
        assert (s_window < l_window)
        assert (15 <= update_interval if isinstance(update_interval, int) else False)
        assert (0 < band_offset if isinstance(band_offset, float) else False)

        EClient.__init__(self, self)
        self.connect('127.0.0.1', 7497, 0)

        self.contract = contract
        self.invest = invest
        self.s_window, self.l_window = s_window, l_window
        self.s_alpha, self.l_alpha = 2 / (s_window + 1), 2 / (l_window + 1)
        self.update_interval = update_interval
        self.band_offset = band_offset

        self.odir = Path('sg3tws_out')
        self.rid = itertools.count()
        self.oid = itertools.count(start=1)

        # init market data subscription
        self.data = {
            'DELAYED_ASK': 0,
            'DELAYED_BID': 0,
            'DELAYED_ASK_SIZE': 0,
            'DELAYED_BID_SIZE': 0,
            'DELAYED_VOLUME': 0,
            'DELAYED_LAST': 0,
            'DELAYED_LAST_SIZE': 0,
            'DELAYED_HIGH': 0,
            'DELAYED_LOW': 0,
            'DELAYED_CLOSE': 0,
            'DELAYED_OPEN': 0
        }

        self.df_data = pd.DataFrame(columns=list(self.data.keys()))
        self.df_data.index.name = 'time'
        self.data_file = self.odir.join('data.csv')
        self.df_data.to_csv(self.data_file)

        self.reqMarketDataType(3)
        self.reqMktData(next(self.rid), self.contract, '', False, False, [])

        # init account data subscription
        self.account = {
            'TotalCashValue': 0.,
            'GrossPositionValue': 0.
        }
        self.df_account = pd.DataFrame(columns=list(self.account.keys()))
        self.df_account.index.name = 'time'
        self.account_file = self.odir.join('account.csv')
        self.df_account.to_csv(self.account_file)
        self.reqAccountSummary(next(self.rid), 'All', ', '.join(self.account.keys()))

        # init trades log
        self.n_order = itertools.count(1)
        self.orders = {
            'time': datetime.datetime.now(),
            'oid': 0,
            'shares': 0,
            'avg_price': 0.,
            'tick_price': 0.,
            'commission': 0.,
            'total': 0.,
            'exchange': ''
        }
        self.df_orders = pd.DataFrame(columns=list(self.orders.keys()))
        self.df_orders.index.name = 'n'
        self.orders_file = self.odir.join('orders.csv')
        self.df_orders.to_csv(self.orders_file)

        # init technical analysis
        self.fields = ['price', 'mal', 'mas', 'mao', 'ub', 'lb', 'cash', 'stock', 'pos', 'total hold', 'total']
        self.stat = dict(zip(self.fields, np.zeros_like(self.fields, dtype=np.float)))

        self.df_stat = pd.DataFrame(columns=self.fields)
        self.df_stat.index.name = 'time'
        self.stat_file = self.odir.join('stat.csv')
        self.df_stat.to_csv(self.stat_file)

        # init plotting
        register_matplotlib_converters()
        self.fig, self.axes = plt.subplots(nrows=4, sharex='all')
        self.fig.set(figwidth=10, figheight=9)
        self.axes[0].set_title('SG3TWS {}'.format(self.contract.symbol), fontsize=18)
        self.axes[0].set_ylabel('Price', fontsize=14)
        self.axes[1].set_ylabel('Oscillator', fontsize=14)
        self.axes[1].axhline(y=0, ls='-', c='k')
        self.axes[2].set_ylabel('Position', fontsize=14)
        self.axes[3].set_ylabel('Cum Returns', fontsize=14)
        self.axes[3].set_xlabel('Time', fontsize=14)
        self.lines = {
            'price': self.axes[0].plot([], [], label='price')[0],
            'mas': self.axes[0].plot([], [], label='mas')[0],
            'mal': self.axes[0].plot([], [], label='mal')[0],
            'mao': self.axes[1].plot([], [], label='mao')[0],
            'ub': self.axes[1].plot([], [], '--k', label='ub')[0],
            'lb': self.axes[1].plot([], [], '--k', label='lb')[0],
            'pos': self.axes[2].plot([], [], label='position')[0],
            'total hold': self.axes[3].plot([], [], label='total hold')[0],
            'total': self.axes[3].plot([], [], label='total')[0],
        }
        for ax in self.axes:
            ax.legend()
        self.fig.tight_layout()

        # initiate TWS-API loop and strategy loop
        threading.Thread(target=self.run, args=()).start()
        threading.Thread(target=self.exec, args=()).start()

    def exec(self):
        time.sleep(2)
        time0 = time.time()

        # init values of first timestamp
        self.append_to_dataframe(self.df_data, list(self.data.values()), self.data_file)
        self.append_to_dataframe(self.df_account, list(self.account.values()), self.account_file)
        self.stat.update({
            'price': self.df_data['DELAYED_LAST'][-1],
            'mas': self.df_data['DELAYED_LAST'][-1],
            'mal': self.df_data['DELAYED_LAST'][-1],
            'ub': self.band_offset,
            'lb': -self.band_offset,
            'cash': self.invest,
            'total hold': self.invest,
            'total': self.invest
        })
        self.append_to_dataframe(self.df_stat, list(self.stat.values()), self.stat_file)

        # sleep till next update
        time.sleep(self.update_interval - (time.time() + time0) % self.update_interval)

        # strategy loop
        while self.isConnected():
            aprint(self.data, fmt='i')

            self.append_to_dataframe(self.df_data, list(self.data.values()), self.data_file)
            self.strategy()
            self.append_to_dataframe(self.df_account, list(self.account.values()), self.account_file)
            self.analyze()
            self.append_to_dataframe(self.df_stat, list(self.stat.values()), self.stat_file)
            self.update_plot()

            # sleep till next update
            time.sleep(self.update_interval - (time.time() + time0) % self.update_interval)

    @staticmethod
    def append_to_dataframe(df, arr, log_file, t=None):
        if t is None:
            df.loc[datetime.datetime.now()] = arr
        else:
            df.loc[t] = arr
        df.iloc[[-1]].to_csv(log_file, mode='a', header=False)

    def strategy(self):
        # analyze time series
        self.stat['price'] = self.df_data['DELAYED_LAST'][-1]
        self.stat['mal'] = self.l_alpha * self.stat['price'] + (1 - self.l_alpha) * self.stat['mal']
        self.stat['mas'] = self.s_alpha * self.stat['price'] + (1 - self.s_alpha) * self.stat['mas']
        self.stat['mao'] = (self.stat['mas'] - self.stat['mal']) / self.stat['mal'] * 100

        # create trade signals
        pos_max = (self.df_stat['cash'][-1] + self.df_stat['pos'][-1] * self.stat['price']) // self.stat['price']
        dpos = 0
        if self.l_window < len(self.df_stat):
            ub_cross = np.diff(np.sign(self.df_stat['mao'][-2:] - self.df_stat['ub'][-2:])).item()
            lb_cross = np.diff(np.sign(self.df_stat['mao'][-2:] - self.df_stat['lb'][-2:])).item()

            # buy if mao crosses ub from below
            if ub_cross > 0:
                dpos = pos_max - self.df_stat['pos'][-1]
            # sell if mao crosses lb from above
            elif lb_cross < 0:
                dpos = -pos_max - self.df_stat['pos'][-1]
            # close if mao crosses ub from above or lb from below
            elif (ub_cross < 0) | (lb_cross > 0):
                dpos = -self.df_stat['pos'][-1]

        # place orders in TWS
        if dpos > 0:
            oid = next(self.oid)
            order = IBOrder(action='BUY', totalQuantity=dpos, orderType='MKT')
            self.placeOrder(oid, self.contract, order)
        elif dpos < 0:
            oid = next(self.oid)
            order = IBOrder(action='SELL', totalQuantity=-dpos, orderType='MKT')
            self.placeOrder(oid, self.contract, order)

    def analyze(self):
        time.sleep(5)
        self.stat['stock'] = self.stat['pos'] * self.stat['price']
        # self.stat['stock'] = self.account['GrossPositionValue'] * np.sign(self.stat['pos'])
        self.stat['total'] = self.stat['cash'] + self.stat['stock']
        pos_hold = self.invest // self.df_data['DELAYED_LAST'][0]
        self.stat['total hold'] = \
            self.df_data['DELAYED_LAST'][-1] * pos_hold + (self.invest - pos_hold * self.df_data['DELAYED_LAST'][0])

    def update_plot(self):
        for k in ['price', 'mas', 'mal', 'mao', 'ub', 'lb', 'total hold', 'total', 'pos']:
            self.lines[k].set_xdata(self.df_stat.index.time)
            self.lines[k].set_ydata(self.df_stat[k].values)
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        self.axes[3].set(
            xticks=np.linspace(*self.axes[0].get_xlim(), 5),
            xticklabels=pd.to_datetime(
                np.linspace(self.df_data.index[0].value, self.df_data.index[-1].value, 5)).strftime('%H:%M:%S')
        )
        self.fig.canvas.draw_idle()

    def error(self, rid, error_code, error_string):
        aprint('id={} / code={} / msg={}'.format(rid, error_code, error_string), fmt='e')

    def tickPrice(self, rid, tick_type, price, attrib):
        self.data[TickTypeEnum.to_str(tick_type)] = price

    def tickSize(self, rid, tick_type, size):
        self.data[TickTypeEnum.to_str(tick_type)] = size

    def execDetails(self, rid, contract, execution):
        self.orders['time'] = datetime.datetime.strptime(execution.time, '%Y%m%d %H:%M:%S')
        self.orders['oid'] = execution.orderId
        self.orders['shares'] = execution.shares if execution.side == 'BOT' else -execution.shares
        self.orders['avg_price'] = execution.avgPrice
        self.orders['tick_price'] = self.df_data['DELAYED_LAST'][-1]
        self.orders['exchange'] = execution.exchange
        aprint(execution.__dict__, fmt='i')

        self.stat['pos'] += self.orders['shares']

    def commissionReport(self, commission_report):
        self.orders['commission'] = commission_report.commission
        self.orders['total'] = self.orders['shares'] * self.orders['avg_price'] + self.orders['commission']
        self.append_to_dataframe(self.df_orders, list(self.orders.values()), self.orders_file, t=next(self.n_order))
        aprint(commission_report.__dict__, fmt='i')

        self.stat['cash'] -= self.orders['total']
        # self.stat['cash'] -= self.orders['shares'] * self.orders['tick_price'] + self.orders['commission']
        # self.stat['cash'] = self.invest + self.df_account['TotalCashValue'][-1] - self.df_account['TotalCashValue'][0]

    def accountSummary(self, rid, account, tag, value, currency):
        self.account[tag] = float(value)


def main():
    from api_util import IBContract
    contract = IBContract(symbol='AAPL', secType='STK', exchange='SMART', currency='USD', primaryExchange='NASDAQ')
    SG3TWS(
        contract=contract,
        invest=10000.,
        s_window=10,
        l_window=30,
        update_interval=30,
        band_offset=0.1
    )


if __name__ == '__main__':
    main()
