from .utils import *
from .plot_all_cutoffs import plot_all_cutoffs

class AllCutoffs:

    def __init__(self, fobject, grid_points=101, fairness_metrics=['acc', 'tpr', 'ppv', 'fpr', 'stp']):
        models = fobject.models
        cutoffs = np.linspace(0, 1, grid_points)
        protected = fobject.protected
        privileged = fobject.privileged

        groups = pd.unique(protected)
        cutoff_data = []

        for i in range(len(models)) :
            for c in cutoffs :
                model = models[i]
                gm = group_matrices(protected, model.y_hat, fobject.y, c, groups=groups)
                gmm = calculate_group_fairness_metrics(gm)
                gmm_loss = calculate_parity_loss(gmm, privileged)
                gmm_loss_unique = gmm_loss[fairness_metrics]

                for m in gmm_loss_unique :
                    cutoff_data.append({
                        "parity_loss": gmm_loss_unique[m][0],
                        "metric": m,
                        "cutoff": c,
                        "label": fobject.label[i]
                    })

        self.cutoff_data = pd.DataFrame(cutoff_data)

    def plot(self):
        return plot_all_cutoffs(self)