import numpy as np
import pandas as pd
from .plotnine import *


def plot_fairobject(fobject, fairness_metrics=['acc', 'tpr', 'ppv', 'fpr', "stp"]):
    n_exp = len(fobject.models)
    data = fobject.fairness_check_data
    data = data[data.metric.isin(fairness_metrics)]
    metrics = pd.unique(fobject.fairness_check_data.metric)
    n_met = len(metrics)
    epsilon = fobject.epsilon

    # bars should start at 0

    data.score -= 1

    upper_bound = max(data.score.max(), 1 / epsilon - 1) + 0.05
    if upper_bound < 0.3:
        upper_bound = 0.3

    lower_bound = min(data.score.min(), epsilon - 1) - 0.05
    if lower_bound > -0.25:
        lower_bound = -0.25

    green = "#c7f5bf"
    red = "#f05a71"

    breaks = np.arange(round(lower_bound, 1), round(upper_bound, 1) + 0.2, 0.2)
    if 0 not in breaks:
        breaks += 0.1
        breaks = breaks.round(1)

    plt = ggplot(data, aes(x="subgroup", y="score", fill="model")) + \
          annotate("rect",
                   xmin=-np.inf,
                   xmax=np.inf,
                   ymin=epsilon - 1,
                   ymax=1 / epsilon - 1,
                   fill=green,
                   alpha=0.1) + \
          annotate("rect",
                   xmin=-np.inf,
                   xmax=np.inf,
                   ymin=-np.inf,
                   ymax=epsilon - 1,
                   fill=red,
                   alpha=0.1) + \
          annotate("rect",
                   xmin=-np.inf,
                   xmax=np.inf,
                   ymin=1 / epsilon - 1,
                   ymax=np.inf,
                   fill=red,
                   alpha=0.1) + \
          geom_bar(stat="identity", position="dodge") + \
          geom_hline(yintercept=0) + \
          coord_flip() + \
          facet_wrap("~metric_name", ncol=1) + \
          scale_y_continuous(limits=(lower_bound, upper_bound),
                             breaks=breaks,
                             labels=breaks + 1) + \
          geom_text(x=0, y=lower_bound - 0.02, label="bias") + \
          ggtitle("Fairness check")

    return plt
