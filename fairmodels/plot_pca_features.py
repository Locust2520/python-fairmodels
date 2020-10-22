import numpy as np
import pandas as pd
from .plotnine import *
from sklearn.decomposition import PCA, FactorAnalysis

def plot_pca_features(X, sensible_features) :

    data = X
    data[data == np.inf] = np.NaN
    data = data.dropna()

    sfindex = [data.columns.get_loc(col) for col in sensible_features]
    print(sfindex)

    # normalisation
    for col in data :
        data[col] -= data[col].min()
        data[col] /= data[col].max()

    pca = PCA(n_components=2)
    pca.fit(data.to_numpy().T)
    fa = FactorAnalysis(n_components=2)
    fa.fit(data.to_numpy().T)

    pca_df = pd.DataFrame(pca.transform(data.to_numpy().T))
    pca_df = pca_df.loc[[col in sensible_features for col in data.columns]]
    # pca_df = pca_df.loc[[data.columns.get_loc(col) for col in sensible_features]]
    pca_df.columns = ["PC1", "PC2"]
    pca_df["x_text"] = pca_df["PC1"] # + 0.1 * np.array([len(f) for f in data.columns])
    pca_df["y_text"] = pca_df["PC2"]
    pca_df["labels"] = data.columns
    exp1, exp2 = pca.explained_variance_ratio_.round(3)

    plt = ggplot() + \
          geom_hline(yintercept=0, color="white", linetype="dashed") + \
          geom_vline(xintercept=0, color="lightgrey", linetype="dashed") + \
          geom_point(data=pca_df, mapping=aes("PC1", "PC2")) + \
          geom_text(data=pca_df,
                    mapping=aes("x_text", "y_text", label="labels"),
                    size=10,
                    color="black") + \
          xlab(f"PCA1 (explained {exp1*100}% variance)") + \
          ylab(f"PCA2 (explained {exp2*100}% variance)") + \
          ggtitle("Features PCA plot")

    return plt