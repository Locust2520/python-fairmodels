from .helper_functions import *
from .plot_ceteris_paribus_cutoff import plot_ceteris_paribus_cutoff

class CeterisParibusCutoff:

    def __init__(self, fobject, subgroup, grid_points=101, fairness_metrics=['acc', 'tpr', 'ppv', 'fpr', 'stp']):
        models = fobject.models
        cutoffs = np.linspace(0, 1, grid_points)
        protected = fobject.protected
        privileged = fobject.privileged

        groups = pd.unique(protected)
        cutoff_data = []
        min_data = []

        for i in range(len(models)) :
            thresholds = {g: models[i].threshold for g in groups}
            min_loss, min_cutoff = np.inf, 0
            for c in cutoffs :
                model = models[i]
                thresholds[subgroup] = c
                gm = group_matrices(protected, model.y_hat, fobject.y, thresholds, groups=groups)
                gmm = calculate_group_fairness_metrics(gm)
                gmm_loss = calculate_parity_loss(gmm, privileged)
                gmm_loss_unique = gmm_loss[fairness_metrics]
                total_loss = gmm_loss_unique.sum(axis=1)[0]
                if total_loss < min_loss :
                    min_loss, min_cutoff = total_loss, c

                for m in gmm_loss_unique :
                    cutoff_data.append({
                        "parity_loss": gmm_loss_unique[m][0],
                        "metric": m,
                        "total_parity_loss": total_loss,
                        "cutoff": c,
                        "model": fobject.label[i]
                    })
            min_data.append({
                "model": model.name,
                "mins": min_cutoff
            })

        self.cutoff_data = pd.DataFrame(cutoff_data)
        self.min_data = pd.DataFrame(min_data)
        self.label = fobject.label
        self.subgroup = subgroup

    def plot(self):
        plot_ceteris_paribus_cutoff(self)
