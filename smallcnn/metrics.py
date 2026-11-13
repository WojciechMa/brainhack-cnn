"""Behavioural effects of the three paradigms from the network's 6-way decisions.

A decision for a class of another task counts as an error; how often that happens is logged
separately per task as "confusion". Next to accuracy, signal detection measures separate
sensitivity (d') from response bias (criterion c > 0 = conservative, < 0 = liberal), so a lesion
that only shifts the network towards one answer does not look like lost information.
When a lesion pushes nearly all trials to one answer, d' from the binary decisions is unreliable;
`separation` then measures what is left from the decision values themselves (AUC, refit threshold).
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, rankdata
from torch.utils.data import DataLoader, Dataset

from data import SET_SIZES, TASKS


@torch.no_grad()
def predict(model, dataset: Dataset, device, batch_size: int = 32) -> np.ndarray:
    """Logits (N, n_classes) in dataset order."""
    model.eval()
    return torch.cat([model(x.to(device)).cpu() for x, _ in DataLoader(dataset, batch_size=batch_size)]).numpy()


def within_task(logits: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Decision between the two classes of each trial's own task only, as in 2-class single-task networks."""
    rows, other = np.arange(len(label)), label ^ 1  # classes come in task pairs (0, 1), (2, 3), (4, 5)
    return np.where(logits[rows, label] > logits[rows, other], label, other)


def dprime(yes_signal: np.ndarray, yes_noise: np.ndarray) -> tuple[float, float]:
    """d' and criterion c from 'yes' answers on signal and noise trials; rates clipped to [1/2n, 1 - 1/2n]."""
    if len(yes_signal) == 0 or len(yes_noise) == 0:
        return np.nan, np.nan
    h = norm.ppf(np.clip(yes_signal.mean(), 0.5 / len(yes_signal), 1 - 0.5 / len(yes_signal)))
    f = norm.ppf(np.clip(yes_noise.mean(), 0.5 / len(yes_noise), 1 - 0.5 / len(yes_noise)))
    return h - f, -(h + f) / 2


def auc(z: np.ndarray, pos: np.ndarray) -> float:
    """Probability that a random positive trial has a higher z than a random negative one (0.5 = no separation)."""
    n1, n0 = pos.sum(), (~pos).sum()
    return (rankdata(z)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def best_threshold(z: np.ndarray, pos: np.ndarray) -> float:
    """Threshold t that maximises the accuracy of 'positive if z > t'."""
    order = np.argsort(z)
    zs, ps = z[order], pos[order]
    correct = np.concatenate([[0], np.cumsum(~ps)]) + ps.sum() - np.concatenate([[0], np.cumsum(ps)])
    edges = np.concatenate([[-np.inf], (zs[:-1] + zs[1:]) / 2, [np.inf]])
    return edges[np.argmax(correct)]


def separation(logits: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    """Per task, from the decision value z = logit(odd class) - logit(even class) of the trial's own task:
    AUC, and accuracy with the threshold refit (2-fold cross-validated, as if the output bias were retrained).
    Per condition: AUC within Posner valid / invalid trials (absent trials matched by cue side, as for d'),
    each search set size and old / new contexts, and the three effects as AUC differences."""
    label = meta.label.to_numpy()
    rows, first = np.arange(len(label)), label & ~1
    z, pos = logits[rows, first + 1] - logits[rows, first], label % 2 == 1
    m = {}
    for t in TASKS:
        s = np.flatnonzero((meta.task == t).to_numpy())
        m[f"{t}_auc"] = auc(z[s], pos[s])
        s = s[np.random.default_rng(0).permutation(len(s))]
        folds = (s[: len(s) // 2], s[len(s) // 2:])
        correct = sum(((z[b] > best_threshold(z[a], pos[a])) == pos[b]).sum() for a, b in (folds, folds[::-1]))
        m[f"{t}_acc_refit"] = correct / len(s)

    p, validity = (meta.task == "posner").to_numpy(), meta.validity.to_numpy()
    target, cue, absent = meta.target.to_numpy(), meta.cue.to_numpy(), p & (validity == "absent")
    for v in ("valid", "invalid"):
        cells = [(p & (validity == v) & (target == side)) | (absent & (cue == (side if v == "valid" else other)))
                 for side, other in (("left", "right"), ("right", "left"))]
        m[f"posner_{v}_auc"] = np.mean([auc(z[s], pos[s]) for s in cells])
    m["posner_cue_effect_auc"] = m["posner_valid_auc"] - m["posner_invalid_auc"]
    for ss in SET_SIZES:
        s = (meta.set_size == ss).to_numpy()
        m[f"search_ss{ss}_auc"] = auc(z[s], pos[s])
    m["search_set_size_effect_auc"] = m[f"search_ss{SET_SIZES[0]}_auc"] - m[f"search_ss{SET_SIZES[-1]}_auc"]
    for c in ("old", "new"):
        s = (meta.config == c).to_numpy()
        m[f"context_{c}_auc"] = auc(z[s], pos[s])
    m["context_effect_auc"] = m["context_old_auc"] - m["context_new_auc"]
    return {k: float(v) for k, v in m.items()}


def task_metrics(pred: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    label = meta.label.to_numpy()
    correct = pred == label
    pred_task = np.asarray(TASKS)[pred // 2]
    m = {"acc": correct.mean()}
    for t in TASKS:
        s = (meta.task == t).to_numpy()
        m[f"{t}_acc"] = correct[s].mean()
        m[f"{t}_confusion"] = (pred_task[s] != t).mean()

    p = (meta.task == "posner").to_numpy()
    for v in ("valid", "invalid"):
        m[f"posner_hit_{v}"] = (pred[p & (meta.validity == v).to_numpy()] == 1).mean()
    m["posner_false_alarm"] = (pred[p & (meta.validity == "absent").to_numpy()] == 1).mean()
    m["posner_cue_effect"] = m["posner_hit_valid"] - m["posner_hit_invalid"]
    # as in the reference Shapley-perturbation code: false alarms from absent trials with the same cue side, then averaged over target sides
    yes, validity, target, cue = pred == 1, meta.validity.to_numpy(), meta.target.to_numpy(), meta.cue.to_numpy()
    absent = p & (validity == "absent")
    m["posner_dprime"], m["posner_criterion"] = dprime(yes[p & ~(validity == "absent")], yes[absent])
    for v in ("valid", "invalid"):
        cells = [(p & (validity == v) & (target == side), absent & (cue == (side if v == "valid" else other)))
                 for side, other in (("left", "right"), ("right", "left"))]
        m[f"posner_dprime_{v}"] = np.mean([dprime(yes[s], yes[n])[0] for s, n in cells])
    m["posner_cue_effect_dprime"] = m["posner_dprime_valid"] - m["posner_dprime_invalid"]

    s_all = (meta.task == "search").to_numpy()
    m["search_dprime"], m["search_criterion"] = dprime(pred[s_all & (label == 3)] == 3, pred[s_all & (label == 2)] == 3)
    for ss in SET_SIZES:
        s = (meta.set_size == ss).to_numpy()
        m[f"search_ss{ss}_acc"] = correct[s].mean()
        m[f"search_ss{ss}_hit"] = (pred[s & (label == 3)] == 3).mean()
        m[f"search_ss{ss}_false_alarm"] = (pred[s & (label == 2)] == 3).mean()
        m[f"search_ss{ss}_dprime"] = dprime(pred[s & (label == 3)] == 3, pred[s & (label == 2)] == 3)[0]
    m["search_set_size_effect"] = m["search_ss1_acc"] - m[f"search_ss{SET_SIZES[-1]}_acc"]
    m["search_set_size_effect_dprime"] = m["search_ss1_dprime"] - m[f"search_ss{SET_SIZES[-1]}_dprime"]

    c_all = (meta.task == "context").to_numpy()
    m["context_dprime"], m["context_criterion"] = dprime(pred[c_all & (label == 5)] == 5, pred[c_all & (label == 4)] == 5)
    for c in ("old", "new"):
        s = (meta.config == c).to_numpy()
        m[f"context_{c}_acc"] = correct[s].mean()
        m[f"context_{c}_dprime"] = dprime(pred[s & (label == 5)] == 5, pred[s & (label == 4)] == 5)[0]
    m["context_effect"] = m["context_old_acc"] - m["context_new_acc"]
    m["context_effect_dprime"] = m["context_old_dprime"] - m["context_new_dprime"]
    return {k: float(v) for k, v in m.items()}


def summary(m: dict[str, float]) -> str:
    ss = "  ".join(f"ss{s} {m[f'search_ss{s}_acc']:.3f}" for s in SET_SIZES)
    return "\n".join([
        f"accuracy {m['acc']:.3f} | per task: " + "  ".join(f"{t} {m[f'{t}_acc']:.3f}" for t in TASKS)
        + " | other-task choices: " + "  ".join(f"{t} {m[f'{t}_confusion']:.3f}" for t in TASKS),
        "d' / criterion: " + "  ".join(f"{t} {m[f'{t}_dprime']:.2f} / {m[f'{t}_criterion']:+.2f}" for t in TASKS),
        f"posner   hit valid {m['posner_hit_valid']:.3f}  invalid {m['posner_hit_invalid']:.3f}"
        f"  false alarms {m['posner_false_alarm']:.3f}  cue effect {m['posner_cue_effect']:+.3f}"
        f" (d' {m['posner_cue_effect_dprime']:+.2f})",
        f"search   accuracy {ss}  set-size effect {m['search_set_size_effect']:+.3f}"
        f" (d' {m['search_set_size_effect_dprime']:+.2f})",
        f"context  old {m['context_old_acc']:.3f}  new {m['context_new_acc']:.3f}  effect {m['context_effect']:+.3f}"
        f" (d' {m['context_effect_dprime']:+.2f})",
    ])
