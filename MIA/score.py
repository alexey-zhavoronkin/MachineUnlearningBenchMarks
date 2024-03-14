import numpy as np
import scipy
from sklearn import metrics


def get_logits(logits: np.ndarray, labels: np.ndarray):
    """Extract numerically stable logits.
    
    Inputs:
        logits: [*, n_classes] - initial logits.
        labels: [*, 1] - correct predictions.
    Returns:
        logits: [*] - logits for ground-truth classes.
    """
    sz = logits.shape[:-1]

    probabilities = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.array(np.exp(probabilities), np.float64)
    probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)

    probabilities = probabilities.reshape(-1, logits.shape[-1])
    y_true = probabilities[np.arange(probabilities.shape[0]), labels.reshape(-1)]
    
    probabilities[np.arange(probabilities.shape[0]), labels.reshape(-1)] = 0
    y_wrong = np.sum(probabilities, axis=-1)

    logit = np.log(y_true+1e-45) - np.log(y_wrong+1e-45)
    logit = logit.reshape(*sz)

    return logit


def sweep(score, x):
    fpr, tpr, _ = metrics.roc_curve(x, -score)
    fnr = 1 - tpr
    tnr = 1 - fpr
    return fnr, tnr, metrics.auc(fnr, tnr)


def lira_offline(target_scores: np.ndarray, shadow_scores: np.ndarray, labels: np.ndarray,
                 fix_variance: bool = False):
    """Score offline using LiRA approach.

    target_scores: [n_examples, n_aug]
    shadow_scores: [n_examples, n_shadow, n_aug]
    labels: [n_examples]    
    """
    mean_out = np.median(shadow_scores, 1)

    if fix_variance:
        std_out = np.std(shadow_scores)
    else:
        std_out = np.std(shadow_scores, 1)
    
    # [n_examples, n_aug], [n_examples, n_aug]
    score = scipy.stats.norm.logpdf(target_scores, mean_out, std_out+1e-30)
    predictions = np.array(score.mean(1))

    fnr, tnr, auc = sweep(np.array(predictions), labels.astype(bool))
    low = tnr[np.where(fnr<0.01)[0][-1]]

    return fnr, tnr, auc, low

def lira_online(target_scores: np.ndarray, shadow_scores: np.ndarray, labels: np.ndarray, in_datasets_list,
                 fix_variance: bool = False):
    """Score offline using LiRA approach.

    target_scores: [n_examples, n_aug]
    shadow_scores: [n_examples, n_shadow, n_aug]
    labels: [n_examples]    
    """

    predictions = []

    for i, in_datasets in enumerate(in_datasets_list):
        mask_in = np.full(shadow_scores.shape[1], False)
        mask_in[in_datasets] = True
        in_shadow_scores = shadow_scores[i, mask_in]
        out_shadow_scores = shadow_scores[i, ~mask_in]
        mean_in = np.median(in_shadow_scores, 0)
        mean_out = np.median(out_shadow_scores, 0)

        if fix_variance:
            std_in = np.std(in_shadow_scores)
            std_out = np.std(out_shadow_scores)
        else:
            std_in = np.std(in_shadow_scores, 0)
            std_out = np.std(out_shadow_scores, 0)

        pr_in = -scipy.stats.norm.logpdf(target_scores[i], mean_in, std_in+1e-30)
        pr_out = -scipy.stats.norm.logpdf(target_scores[i], mean_out, std_out+1e-30)
        score = pr_in-pr_out
        predictions.append(score.mean())

    fnr, tnr, auc = sweep(np.array(predictions), labels.astype(bool))
    low = np.max(tnr[np.where(fnr<0.01)[0]])

    return fnr, tnr, auc, low