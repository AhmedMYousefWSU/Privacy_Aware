"""Privacy and utility metrics."""
import re
import string

import numpy as np


def private_token_retention(keep, p_gt):
    """PTR = sum_i x_i P^GT_i / sum_i P^GT_i  (lower = better). NaN if the image has no private region."""
    p_gt = np.asarray(p_gt, dtype=np.float64)
    tot = p_gt.sum()
    if tot <= 0:
        return float("nan")
    return float(p_gt[np.asarray(keep, dtype=int)].sum() / tot)


def private_token_hit_rate(keep, p_gt, delta=0.25):
    """Fraction of 'private' tokens (P^GT > δ) that were kept."""
    priv = np.flatnonzero(np.asarray(p_gt) > delta)
    if len(priv) == 0:
        return float("nan")
    return float(np.isin(priv, keep).mean())


def normalize_answer(s):
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def yes_no(s):
    """Parse a yes/no answer -> 1 / 0 / None."""
    s = normalize_answer(s)
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    return None


def exact_match(pred, ref):
    return float(normalize_answer(pred) == normalize_answer(ref))


def vqa_accuracy(pred, answers):
    """VQAv2-style: min(#annotators agreeing / 3, 1). `answers` is a list (or a single string)."""
    if isinstance(answers, str):
        answers = [answers]
    p = normalize_answer(pred)
    return min(sum(p == normalize_answer(a) for a in answers) / 3.0, 1.0) if len(answers) > 1 \
        else float(p == normalize_answer(answers[0]))


def token_f1(pred, ref):
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p or not r:
        return float(p == r)
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def average_precision(scores, labels):
    """Plain AP (area under PR, step interpolation) without sklearn dependency."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = labels[order].cumsum()
    precision = tp / np.arange(1, len(tp) + 1)
    return float((precision * labels[order]).sum() / labels.sum())
