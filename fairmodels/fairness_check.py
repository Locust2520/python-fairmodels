import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from .plot_fairness_object import plot_fairobject
from .plot_density import plot_density
from .all_cutoffs import AllCutoffs
from .ceteris_paribus_cutoff import CeterisParibusCutoff


class ModelProb:
    n = 0

    def __init__(self, preds, threshold, name=None):
        ModelProb.n += 1
        self.preds = np.array(preds)
        self.y_hat = self.preds
        self.threshold = threshold
        if name is not None:
            self.name = name
        else:
            self.name = f"Model {ModelProb.n}"


class FairMetric:
    def __init__(self, function, short_name, long_name=None):
        self.function = function
        self.short_name = short_name
        if long_name is None:
            self.long_name = short_name
        else:
            self.long_name = long_name


def basic_metrics():
    # Definition of basic fair metrics
    def tpr(tp, tn, fp, fn):
        return tp / (tp + fn)

    def tnr(tp, tn, fp, fn):
        return tn / (tn + fp)

    def ppv(tp, tn, fp, fn):
        return tp / (tp + fp)

    def npv(tp, tn, fp, fn):
        return tn / (tn + fn)

    def fnr(tp, tn, fp, fn):
        return fn / (fn + tp)

    def fpr(tp, tn, fp, fn):
        return fp / (fp + tn)

    def fdr(tp, tn, fp, fn):
        return fp / (fp + tp)

    def for_(tp, tn, fp, fn):
        return fn / (fn + tn)

    def ts(tp, tn, fp, fn):
        return tp / (tp + fn + fp)

    def stp(tp, tn, fp, fn):
        return (tp + fp) / (tp + fn + fp + tn)

    def acc(tp, tn, fp, fn):
        return (tp + tn) / (tp + fn + fp + tn)

    def f1(tp, tn, fp, fn):
        ppv_ = tp / (tp + fp)
        tpr_ = tp / (tp + fn)
        return 2 * ppv_ * tpr_ / (ppv_ + tpr_)

    return [
        FairMetric(tpr, "tpr", "Equal opportunity ratio     TP/(TP + FN)"),
        FairMetric(tnr, "tnr"),
        FairMetric(ppv, "ppv", "Predictive parity ratio     TP/(TP + FP)"),
        FairMetric(npv, "npv"),
        FairMetric(fnr, "fnr"),
        FairMetric(fpr, "fpr", "Predictive equality ratio   FP/(FP + TN)"),
        FairMetric(fdr, "fdr"),
        FairMetric(for_, "for"),
        FairMetric(ts , "ts"),
        FairMetric(stp, "stp", "Statistical parity ratio   (TP + FP)/(TP + FP + TN + FN)"),
        FairMetric(acc, "acc", "Accuracy equality ratio    (TP + TN)/(TP + FP + TN + FN)"),
        FairMetric(f1 , "f1")
    ]


fair_metrics = basic_metrics()


class FairnessObject:
    def __init__(self, model_probs, y, protected, privileged, add_metrics=[], epsilon=0.8):
        # add_metric: list FairMetric objects
        all_metrics = [(m.short_name, m.long_name, m.function) for m in fair_metrics + add_metrics]
        groups = np.unique(protected)
        parity_loss_metric_data = pd.DataFrame()
        groups_data = dict()
        groups_confusion_matrices = dict()

        for model in model_probs :
            groups_data[model.name] = dict()
            groups_confusion_matrices[model.name] = dict()

            # ~~~~~~~~~~~ scores for each group ~~~~~~~~~~~~~ #
            for group in groups :
                # we only care about the instances of a specific group (= class)
                g_preds = model.preds[protected == group]
                g_y = y[protected == group]
                # computing basic metrics
                tn, fp, fn, tp = confusion_matrix(g_y, g_preds > model.threshold).flatten()
                scores = {m: function(tp, tn, fp, fn) for m, _, function in all_metrics}

                # converting to a dataframe
                scores = pd.DataFrame({k: [scores[k]] for k in scores})
                groups_data[model.name][group] = scores

                # same thing for the confusion matrix
                cm = pd.DataFrame({
                    "tn": [tn],
                    "fp": [fp],
                    "fn": [fn],
                    "tp": [tp]
                })
                groups_confusion_matrices[model.name][group] = cm

            # ~~~~~~~~~~~~~~~ model loss ~~~~~~~~~~~~~~~ #
            all_groups_scores = groups_data[model.name].values()
            privileged_scores = groups_data[model.name][privileged]

            # just making a dataframe filled with zeros
            loss = privileged_scores - privileged_scores
            # then do a sum
            for scores in all_groups_scores:
                loss += np.abs(np.log(scores / privileged_scores))

            parity_loss_metric_data = parity_loss_metric_data.append(loss)

        # ~~~~~~~~~~~~~~~ fairness check data ~~~~~~~~~~~~~~~ #
        fairness_check_data = []

        for model in model_probs :
            mn = model.name
            for subgroup in groups_data[mn] :
                if subgroup == privileged :
                    continue
                for metric, metric_name, _ in all_metrics :
                    fairness_check_data.append({
                        "score": groups_data[mn][subgroup][metric][0] / groups_data[mn][privileged][metric][0],
                        "subgroup": subgroup,
                        "metric_name": metric_name,
                        "metric": metric,
                        "model": mn
                    })
        fairness_check_data = pd.DataFrame(fairness_check_data)

        self.y = y
        self.parity_loss_metric_data = parity_loss_metric_data
        self.groups_data = groups_data
        self.groups_confusion_matrices = groups_confusion_matrices
        self.fairness_check_data = fairness_check_data
        self.models = model_probs
        self.privileged = privileged
        self.protected = protected
        self.label = [m.name for m in model_probs]
        self.epsilon = epsilon
        self.all_cutoffs = None
        self.ceteris_paribus_cutoff = None

    def plot(self, **kwargs):
        return plot_fairobject(self, **kwargs)

    def plot_density(self, **kwargs):
        return plot_density(self, **kwargs)

    def plot_all_cutoffs(self,
                         grid_points=101,
                         fairness_metrics=['acc', 'tpr', 'ppv', 'fpr', 'stp']):
        if self.all_cutoffs is None :
            self.all_cutoffs = AllCutoffs(self, grid_points, fairness_metrics)
        return self.all_cutoffs.plot()

    def plot_ceteris_paribus_cutoff(self,
                                    subgroup,
                                    grid_points=101,
                                    fairness_metrics=['acc', 'tpr', 'ppv', 'fpr', 'stp']):
        if self.ceteris_paribus_cutoff is None:
            self.ceteris_paribus_cutoff = CeterisParibusCutoff(self, subgroup, grid_points, fairness_metrics)
        return self.ceteris_paribus_cutoff.plot()

    def __plot__(self, **kwargs):
        return plot_fairobject(self, **kwargs)


def fairness_check(**kwargs) :
    return FairnessObject(**kwargs)