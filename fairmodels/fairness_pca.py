import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class FairnessPCA:

    def __init__(self, fobject, fairness_metrics=['acc', 'tpr', 'ppv', 'fpr', 'stp']):

        data = fobject.parity_loss_metric_data[fairness_metrics]
        data[data == np.inf] = np.NaN
        data = data.dropna(axis=1)

        pca = PCA(n_components=2)
        pca.fit(data.to_numpy())

        x = pd.DataFrame(pca.transform(data.to_numpy()))
        x.columns = ["PC1", "PC2"]
        x["labels"] = data.index
        d = x["PC1"].abs().max() + x["PC2"].abs().max()
        # rotation = pd.DataFrame(pca.transform(np.identity(len(data.columns))))
        rotation = pd.DataFrame(pca.components_.T) * d
        rotation.columns = ["PC1", "PC2"]
        rotation["labels"] = data.columns

        self.pc_1_2 = pca.explained_variance_ratio_.round(2)
        self.rotation = rotation
        self.x = x
        self.sdev = pca.singular_values_
        self.label = fobject.label