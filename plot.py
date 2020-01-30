from typing import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def plot_column(df, name, ax, style=None):
    if name in df:
        if style is None:
            ax.plot(df[name].fillna(0).values, label=name)
        else:
            ax.plot(df[name].fillna(0).values, style, label=name)


def plot_bt(
        df: pd.DataFrame,
        title: AnyStr = ''
) -> matplotlib.figure.Figure:

    fig, axes = plt.subplots(nrows=3, sharex='all')
    fig.set(figwidth=10, figheight=9)

    plot_column(df, 'Price', axes[0])
    plot_column(df, 'MAS', axes[0])
    plot_column(df, 'MAL', axes[0])
    axes[0].set_title(title, fontsize=18)
    axes[0].set_ylabel('Price', fontsize=14)

    plot_column(df, 'Oscillator', axes[1])
    plot_column(df, 'UB', axes[1], style='--k')
    plot_column(df, 'LB', axes[1], style='--k')
    plot_column(df, 'UB2', axes[1], style='--k')
    plot_column(df, 'LB2', axes[1], style='--k')
    axes[1].set_ylabel('Oscillator', fontsize=14)
    if 'UB' in df:
        axes[1].set(ylim=(-4 * df['UB'].max(), 4 * df['UB'].max()))
    axes[1].axhline(y=0, ls='-', c='k')

    plot_column(df, 'Cum Returns Hold', axes[2])
    plot_column(df, 'Cum Returns Gross', axes[2])
    plot_column(df, 'Cum Returns Net', axes[2])
    axes[2].set_ylabel('Cum Returns', fontsize=14)
    axes[2].set_xlabel('Time', fontsize=14)

    xlabels_mask = df.index.to_series().groupby(pd.Grouper(freq='D')).apply(lambda t: t == t.min()).values
    for ax in axes:
        if 'Positions' in df:
            ax.pcolorfast((0, len(df)), ax.get_ylim(), df['Positions'].values[None], cmap='RdYlGn', alpha=0.3)
        ax.legend()
        ax.set(
            xticks=np.arange(len(df))[xlabels_mask],
            xticklabels=df.index[xlabels_mask].strftime('%d/%m/%Y')
        )
        ax.xaxis.set_tick_params(rotation=45)
        ax.xaxis.grid(True, linestyle=':', linewidth=2)

    fig.tight_layout()
    return fig


class CompPlot:
    def __init__(
            self,
            title: AnyStr = '',
            ylabel: AnyStr = ''
    ) -> None:

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title, fontsize=18)
        self.ax.set_ylabel(ylabel, fontsize=14)
        self.ax.set_xlabel('Time', fontsize=14)

    def plot(
            self,
            x: pd.Series,
            style: Optional[AnyStr] = None,
            label: AnyStr = ''
    ) -> None:
        if style is None:
            self.ax.plot(x.values, label=label)
        else:
            self.ax.plot(x.values, style, label=label)
        xlabels_mask = x.index.to_series().groupby(pd.Grouper(freq='D')).apply(lambda t: t == t.min()).values
        self.ax.set(
            xticks=np.arange(len(x))[xlabels_mask],
            xticklabels=x.index[xlabels_mask].strftime('%d/%m/%Y')
        )
        self.ax.xaxis.set_tick_params(rotation=45)
        self.ax.xaxis.grid(True, linestyle=':', linewidth=2)

    def post_process(self) -> matplotlib.figure.Figure:
        self.ax.legend()
        self.fig.tight_layout()
        return self.fig
