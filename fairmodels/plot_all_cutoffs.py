import numpy as np
import pandas as pd
from .plotnine import *


def plot_all_cutoffs(all_cutoffs):
    data = all_cutoffs.cutoff_data

    plt = ggplot(data, aes("cutoff", "parity_loss", color="metric")) + \
        geom_line() + \
        labs(color="parity loss metric") + \
        ggtitle("All cutoffs plot") + \
        facet_wrap("~label") + \
        xlab("value of cutoff") + \
        ylab("Metric's parity loss")

    return plt