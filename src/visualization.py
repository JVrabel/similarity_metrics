import numpy as np
import plotly.graph_objects as go
from enum import Enum, auto
from typing import TypeVar, Tuple, Iterable, Optional, Callable
from plotly.express import colors


def plot_spectra(spectra: np.ndarray,
                 wave: Optional[Iterable]=None,
                 title: Optional[str]=None,
                 labels: Optional[Iterable[str]]=None,
                 colormap=colors.qualitative.Set1,
                 axes_titles: bool=True,
                 opacity: float = .7,
                 ):
    if wave is None:
        wave = np.arange(len(spectra[0]))
    if labels is None:
        labels = ["class {}".format(x+1) for x in range(len(spectra))]
    fig = go.Figure()
    for i in range(len(spectra)):
        fig.add_trace(
            go.Scatter(
                x = wave,
                y = spectra[i],
                name = str(labels[i]),
                line = {'color': colormap[i % len(colormap)]},
                opacity=opacity,
            )
        )
    fig.update_layout(
        title = title,
        xaxis_title = "wavelengths (nm)" if axes_titles else "",
        yaxis_title = "relative intensity (-)" if axes_titles else "")
    return fig


def plot_one_spectrum(spectra: np.ndarray,
                 wave: Optional[Iterable]=None,
                 title: Optional[str]=None,
                 labels: Optional[Iterable[str]]=None,
                 colormap=colors.qualitative.Set1,
                 axes_titles: bool=True,
                 opacity: float = .7,
                 ):
    if wave is None:
        wave = np.arange(len(spectra[0]))
    if labels is None:
        labels = ["class {}".format(x+1) for x in range(len(spectra))]
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x = wave,
            y = spectra,
            name = str(labels),
            opacity=opacity,
        )
    )
    
    fig.update_layout(
        title = title,
        xaxis_title = "wavelengths (nm)" if axes_titles else "",
        yaxis_title = "relative intensity (-)" if axes_titles else "")
    return fig
