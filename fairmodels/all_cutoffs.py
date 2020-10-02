from .helper_functions import *

class AllCutoffs:

    def __init__(self, fobject, grid_points=101, fairness_metrics=['acc', 'tpr', 'ppv', 'fpr', 'stp']):
        models = fobject.models
        cutoffs = np.linspace(0, 1, grid_points)
        protected = fobject.protected
        privileged = fobject.privileged

        groups = pd.unique(protected)
        n_subgroups = len(groups)
        cutoff_data = []

        for i in range(len(models)) :
            for c in cutoffs :
                model = models[i]
                gm = group_matrices(protected, model.y_hat, fobject.y, c)
                gmm = calculate_group_fairness_metrics(gm)
                gmm_loss = calculate_parity_loss(gmm, privileged)
                gmm_loss_unique = gmm_loss[fairness_metrics]

                for m in gmm_loss_unique :
                    cutoff_data.append({
                        "partity_loss": gmm_loss_unique[m][0],
                        "metric": m,
                        "cutoff": c,
                        "label": fobject.label[i]
                    })

        self.cutoff_data = pd.DataFrame(cutoff_data)