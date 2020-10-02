import numpy as np
import pandas as pd
from .plotnine import *


def plot_ceteris_paribus_cutoff(cpc):
    data = cpc.cutoff_data
    models = cpc.label
    n_models = len(models)

    min_data = cpc.min_data
    min_data["y"] = [np.min(data.parity_loss)] * n_models
    min_data["tyend"] = [np.max(data.parity_loss)] * n_models
    min_data["yend"] = min_data.tyend * 0.95
    min_data.y -= min_data.tyend / 20

    plt = ggplot(data, aes("cutoff", "parity_loss", color="metric")) + \
        geom_line() + \
        labs(color="parity loss metric") + \
        facet_wrap("~model") + \
        ggtitle("Ceteris paribus cutoff plot") + \
        xlab("value of cutoff") + \
        ylab("Metric's parity loss") + \
        geom_segment(data=min_data, mapping=aes(x="mins", xend="mins", y="y", yend="yend"),
                     linetype="dashed", color="grey") + \
        geom_text(data=min_data, mapping=aes(x="mins", y="tyend", label="mins"), size=10, color="grey")

    return plt