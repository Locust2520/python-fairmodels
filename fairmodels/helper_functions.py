import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def group_matrices(protected, probs, y, cutoff) :
    gm = dict()
    groups = np.unique(protected)
    for group in groups:
        # we only care about the instances of a specific group (= class)
        g_preds = probs[protected == group]
        g_y = y[protected == group]
        # computing basic metrics
        tn, fp, fn, tp = confusion_matrix(g_y, g_preds > cutoff).flatten()
        gm[group] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
    return gm


def calculate_group_fairness_metrics(gm) :
    metrics = dict()
    for group in gm :
        tp = gm[group]["tp"]
        tn = gm[group]["tn"]
        fp = gm[group]["fp"]
        fn = gm[group]["fn"]
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
        scores = pd.DataFrame([scores])
        metrics[group] = scores
    return metrics


def calculate_parity_loss(gmm, privileged) :
    all_groups_scores = gmm.values()
    privileged_scores = gmm[privileged]

    # just making a dataframe filled with zeros
    loss = privileged_scores - privileged_scores
    # then do a sum
    for scores in all_groups_scores:
        loss += np.abs(np.log(scores / privileged_scores))

    return loss