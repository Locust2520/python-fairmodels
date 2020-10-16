from .plotnine import *
import pandas as pd
import numpy as np

def plot_fairness_heatmap(fobject, fairness_metrics=None):

    data = fobject.parity_loss_metric_data
    if fairness_metrics is not None :
        data = data[fairness_metrics]
    data = data.round(2)
    data[data == np.inf] = np.NaN
    midpoint = data.max().max() / 1.5

    expanded = []
    for metric in data.columns :
        for model in data.index :
            expanded.append({
                "parity_loss_metric": metric,
                "model": model,
                "score": data[metric].loc[model]
            })
    heatmap_data = pd.DataFrame(expanded)

    plt = ggplot(heatmap_data, aes("parity_loss_metric", "model", fill="score")) + \
          geom_tile(colour="white", size=2, na_rm=True) + \
          scale_fill_gradient2(low="#c7f5bf",
                               mid="#46bac2",
                               high="#371ea3",
                               midpoint=midpoint,
                               na_value="lightgrey") + \
          geom_text(aes(label="score"), size=9, fontweight=512) + \
          ggtitle("Fairness Heatmap")

    return plt