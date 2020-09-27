import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from fairmodels.plot_fairness_object import plot_fairobject


class ModelProb:
    n = 0

    def __init__(self, preds, threshold, name=None):
        ModelProb.n += 1
        self.preds = np.array(preds)
        self.y_hat = self.preds
        self.threshold = threshold
        if name:
            self.name = name
        else:
            self.name = f"Model {ModelProb.n}"


class FairnessObject:
    def __init__(self, model_probs, y, protected, privileged, epsilon=0.8):
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
                scores = {
                    "tpr": tp / (tp + fn),
                    "tnr": tn / (tn + fp),
                    "ppv": tp / (tp + fp),
                    "npv": tn / (tn + fn),
                    "fnr": fn / (fn + tp),
                    "fpr": fp / (fp + tn),
                    "fdr": fp / (fp + tp),
                    "for": fn / (fn + tn),
                    "ts": tp / (tp + fn + fp),
                    "stp": (tp + fp) / (tp + fn + fp + tn),
                    "acc": (tp + tn) / (tp + fn + fp + tn)
                }
                scores["f1"] = 2 * scores["ppv"] * scores["tpr"] / (scores["ppv"] + scores["tpr"])

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
        metrics = [
            ("acc", "Accuracy equality ratio    (TP + TN)/(TP + FP + TN + FN)"),
            ("ppv", "Predictive parity ratio     TP/(TP + FP)"),
            ("fpr", "Predictive equality ratio   FP/(FP + TN)"),
            ("tpr", "Equal opportynity ratio     TP/(TP + FN)"),
            ("stp", "Statistical parity ratio   (TP + FP)/(TP + FP + TN + FN)")]

        for model in model_probs :
            mn = model.name
            for subgroup in groups_data[mn] :
                if subgroup == privileged :
                    continue
                for metric, metric_name in metrics :
                    fairness_check_data.append({
                        "score": groups_data[mn][subgroup][metric][0] / groups_data[mn][privileged][metric][0],
                        "subgroup": subgroup,
                        "metric": metric_name,
                        "model": mn
                    })
        fairness_check_data = pd.DataFrame(fairness_check_data)

        self.parity_loss_metric_data = parity_loss_metric_data
        self.groups_data = groups_data
        self.groups_confusion_matrices = groups_confusion_matrices
        self.fairness_check_data = fairness_check_data
        self.models = model_probs
        self.privileged = privileged
        self.protected = protected
        self.label = [m.name for m in model_probs]
        self.epsilon = epsilon

    def plot(self):
        return plot_fairobject(self)

    def __plot__(self):
        return plot_fairobject(self)


def fairness_check(**kwargs) :
    return FairnessObject(**kwargs)