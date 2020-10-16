from .plotnine import *

def plot_fairness_pca(fpca) :

    pca_data = fpca.x
    pc_1_2 = fpca.pc_1_2
    pca_feature = fpca.rotation
    n = len(pca_feature)

    pca_data["x_text"] = pca_data["PC1"] * 1.1
    pca_data["y_text"] = pca_data["PC2"] * 1.1
    pca_feature["x_text"] = pca_feature["PC1"] * 1.1
    pca_feature["y_text"] = pca_feature["PC2"] * 1.1

    lab_x = f"PC1: explained {pc_1_2[0]*100}% of variance"
    lab_y = f"PC2: explained {pc_1_2[1]*100}% of variance"

    plt = ggplot() + \
          geom_hline(yintercept=0, color="white", linetype="dashed") + \
          geom_vline(xintercept=0, color="lightgrey", linetype="dashed") + \
          geom_segment(data=pca_feature,
                       mapping=aes(
                           x=[0]*n,
                           y=[0]*n,
                           xend="PC1",
                           yend="PC2"),
                       color="red",
                       alpha=0.5) + \
          geom_text(data=pca_feature,
                    mapping=aes(x="x_text",
                                y="y_text",
                                label="labels"),
                    color="red", alpha=0.5, size=8) + \
          geom_text(data=pca_data,
                    mapping=aes("x_text", "y_text", label="labels"),
                    size=10,
                    color="black") + \
          geom_point(data=pca_data, mapping=aes("PC1", "PC2")) + \
          xlab(lab_x) + \
          ylab(lab_y) + \
          ggtitle("Fairness PCA plot")

    return plt