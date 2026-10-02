import re
from collections import Counter
import numpy as np


def classification(labels, predictions, classes):
    labels, predictions = np.asarray(labels, dtype=np.int64), np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or (labels < 0).any() or (labels >= classes).any() or (predictions < 0).any() or (predictions >= classes).any():
        raise ValueError("Invalid labels, predictions or class vocabulary")
    confusion = np.bincount(labels*classes+predictions, minlength=classes*classes).reshape(classes, classes)
    tp = confusion.diagonal().astype(float)
    support, predicted = confusion.sum(1), confusion.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=precision+recall > 0)
    observed = support > 0
    return {"accuracy": float(tp.sum()/max(1, confusion.sum())),
            "macro_f1": float(f1.mean()),  # fixed configured class vocabulary, including zero support
            "balanced_accuracy": float(recall[observed].mean()) if observed.any() else 0.,
            "per_class": [{"class": i, "precision": float(precision[i]), "recall": float(recall[i]),
                           "f1": float(f1[i]), "support": int(support[i])} for i in range(classes)],
            "confusion": confusion.tolist()}


def participant_bootstrap(labels, predictions, participants, classes, seed=0, repeats=1000):
    """Resample people, never overlapping windows as independent replicates."""
    ids = sorted(set(participants))
    if len(ids) < 2:
        return {"macro_f1_95ci": None, "participants": len(ids), "reason": "At least two test participants required"}
    rng = np.random.default_rng(seed)
    groups = {p: np.flatnonzero(np.asarray(participants) == p) for p in ids}
    values = []
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    for _ in range(repeats):
        indices = np.concatenate([groups[p] for p in rng.choice(ids, len(ids), replace=True)])
        values.append(classification(labels[indices], predictions[indices], classes)["macro_f1"])
    return {"macro_f1_95ci": np.quantile(values, [.025, .975]).tolist(), "participants": len(ids)}


def paired_participant_test(labels, first, second, participants, classes, seed=0, repeats=10000):
    ids = sorted(set(participants))
    labels, first, second = map(np.asarray, (labels, first, second))
    deltas = []
    for p in ids:
        keep = np.asarray(participants) == p
        deltas.append(classification(labels[keep], first[keep], classes)["macro_f1"] -
                      classification(labels[keep], second[keep], classes)["macro_f1"])
    deltas = np.asarray(deltas)
    if len(ids) < 2:
        return {"p_value": None, "participants": len(ids)}
    rng = np.random.default_rng(seed)
    simulated = (rng.choice([-1, 1], (repeats, len(ids)))*deltas).mean(1)
    p = (1+np.count_nonzero(np.abs(simulated) >= abs(deltas.mean())-1e-12))/(repeats+1)
    return {"p_value": float(p), "mean_participant_f1_difference": float(deltas.mean()), "participants": len(ids)}


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    result, running = [0.]*len(order), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., p_values[i]*(len(order)-rank)))
        result[i] = running
    return result


def answer_metrics(reference, prediction):
    def tokens(text):
        return re.findall(r"\w+", text.lower())
    a, b = tokens(reference), tokens(prediction)
    overlap = sum((Counter(a)&Counter(b)).values())
    f1 = 2*overlap / max(1, len(a)+len(b))
    return {"exact_match": float(a == b), "token_f1": f1 if a or b else 1.}
