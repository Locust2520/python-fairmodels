import numpy as np
import pandas as pd
from .plotnine import *


def plot_density(fobject) :

    models = fobject.models
    m = len(fobject.protected)
    density_data = pd.DataFrame()

    for i in range(len(models)) :
        tmp_data = pd.DataFrame({
            "probability": models[i].y_hat,
            "label": [fobject.label[i]] * m,
            "protected": fobject.protected
        })
        # bind with rest
        density_data = density_data.append(tmp_data)

    print(density_data)

    plt = ggplot(density_data, aes(x='factor(protected)', y='probability')) + \
        geom_violin(color="#ceced9", fill="#ceced9", alpha=0.5) + \
        geom_boxplot(aes(fill="protected"), width=0.3, alpha=0.5, outlier_alpha=0) + \
        scale_y_continuous(limits=(0, 1)) + \
        xlab("protected variable") + \
        coord_flip() + \
        facet_grid("~label") + \
        ggtitle("Density plot")

    return plt