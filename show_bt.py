import datetime
import numpy as np
import pandas as pd

from util import Path
from datafetcher import DataFetcher
from backtest import SG1, SG2, SG3, SG4, SG5, SG6
from plot import plot_bt, CompPlot


odir = Path('out_bt', replace=True)

# obtain market data
symbol = 'AAPL'
start = datetime.datetime.combine(datetime.date.today(), datetime.time(0)) - datetime.timedelta(days=7)
end = datetime.datetime.now()
asset = DataFetcher.get_intraday_asset_data(symbol, start, end).ffill()
print(asset.columns)

# specify strategies
strategies = [
    SG1(
        price=asset['close'],
        fees=0.0005,
        oscillator='RSIN',
        window=60,
        ewm=True
    ),
    SG2(
        price=asset['close'],
        fees=0.0005,
        oscillator='RSIN',
        window=60,
        band_alpha=0.2,
        ewm=True
    ),
    SG3(
        price=asset['close'],
        fees=0.0005,
        oscillator='RSIN',
        window=60,
        band_alpha=0.2,
        ewm=True
    ),
    SG4(
        price=asset['close'],
        fees=0.0005,
        oscillator='RSIN',
        window=60,
        band_alpha=0.2,
        ewm=True
    ),
    SG5(
        price=asset['close'],
        fees=0.0005,
        oscillator='RSIN',
        window=60,
        band_alpha=0.2,
        ewm=True
    ),
    SG6(
        price=asset['close'],
        fees=0.0005,
        oscillator='RSIN',
        window=60,
        band_alpha=0.2,
        band_alpha2=0.1,
        ewm=True
    )
]

# plot and evaluate
metrics = pd.DataFrame(
    columns=['Mean Returns Net', 'Mean Returns Gross', 'Mean Returns Hold', 'Cum Returns Net', 'Cum Returns Gross',
             'Cum Returns Hold', 'Num Trades']
)
cp = CompPlot(title=symbol, ylabel='Cum Returns')
for sg in strategies:
    title = '{}_{}_{}'.format(type(sg).__name__, sg.oscillator, symbol)
    metrics.loc[title] = np.r_[sg.mean_returns().values, sg.cum_returns().values, sg.num_trades()]

    fig = plot_bt(sg.df, title)
    fig.show()
    fig.savefig(odir.join(title + '.png'), dpi=250)

    cp.plot(sg.df['Cum Returns Net'], label=title + ' Net')

metrics.to_csv(odir.join('metrics.csv'))
cp.plot(sg.df['Cum Returns Hold'], 'k--', label='Hold')
fig = cp.post_process()
fig.show()
fig.savefig(odir.join('cum_returns_net.png'), dpi=250)
